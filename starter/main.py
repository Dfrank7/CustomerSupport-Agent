"""
Customer Support AI Agent — Complete Implementation
===================================================
Implements AgentCore Runtime, MCP Gateway integration, Bedrock Knowledge Base RAG,
AgentCore Long-Term Memory hooks, Code Interpreter loyalty calculator, and Browser tool.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-vkd3cpekid.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "N2XAHKP56K"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-qI2h6LGEVl"


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    namespaces_map = {}
    try:
        strategies = mem_client.get_memory_strategies(memory_id)
        if isinstance(strategies, dict):
            strategies = strategies.get("memoryStrategies", strategies.get("strategies", []))
        for strategy in strategies:
            strat_type = strategy.get("type") or strategy.get("memoryStrategyType") or strategy.get("name", "DEFAULT")
            templates = strategy.get("namespaceTemplates") or strategy.get("namespaces") or []
            if templates:
                namespaces_map[strat_type] = templates[0]
    except Exception as e:
        logger.warning(f"Failed to fetch memory strategies: {e}")
    return namespaces_map


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(self.memory_client, self.memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        if not event.agent.messages:
            return
        last_msg = event.agent.messages[-1]
        if last_msg.get("role") != "user":
            return

        content = last_msg.get("content", [])
        if not isinstance(content, list) or not content:
            return

        # Skip toolResult messages; only run on plain-text user messages
        if any(isinstance(block, dict) and "toolResult" in block for block in content):
            return

        text_block_idx = None
        user_query = ""
        for idx, block in enumerate(content):
            if isinstance(block, dict) and "text" in block:
                text_block_idx = idx
                user_query = block["text"]
                break

        if not user_query or text_block_idx is None:
            return

        collected_memories = []
        for strat_type, ns_template in self.namespaces.items():
            namespace = ns_template.format(actorId=self.actor_id)
            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=user_query,
                    top_k=5,
                )
                if isinstance(memories, dict):
                    memories = memories.get("memories", memories.get("memoryRecords", []))
                for mem in memories:
                    if isinstance(mem, dict):
                        mem_content = mem.get("content", mem.get("text", ""))
                        if isinstance(mem_content, dict):
                            mem_text = mem_content.get("text", "")
                        else:
                            mem_text = str(mem_content)
                    else:
                        mem_text = str(mem)
                    mem_text = mem_text.strip()
                    if mem_text:
                        collected_memories.append(f"[{strat_type}] {mem_text}")
            except Exception as e:
                logger.warning(f"Error retrieving memories for {namespace}: {e}")

        if collected_memories:
            memories_str = "\n".join(collected_memories)
            content[text_block_idx]["text"] = (
                f"Customer Context:\n{memories_str}\n\n{user_query}"
            )

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages
        customer_query = None
        agent_response = None

        for msg in reversed(messages):
            role = msg.get("role")
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            if role == "assistant" and agent_response is None:
                texts = [
                    b["text"]
                    for b in content
                    if isinstance(b, dict) and "text" in b and b["text"].strip()
                ]
                if texts:
                    agent_response = "\n".join(texts)

            elif role == "user" and customer_query is None:
                if any(isinstance(b, dict) and "toolResult" in b for b in content):
                    continue
                texts = [
                    b["text"]
                    for b in content
                    if isinstance(b, dict) and "text" in b and b["text"].strip()
                ]
                if texts:
                    raw_text = "\n".join(texts)
                    if raw_text.startswith("Customer Context:\n") and "\n\n" in raw_text:
                        customer_query = raw_text.split("\n\n", 1)[1]
                    else:
                        customer_query = raw_text

            if customer_query and agent_response:
                break

        if customer_query and agent_response:
            try:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[
                        (customer_query, "USER"),
                        (agent_response, "ASSISTANT"),
                    ],
                )
            except Exception as e:
                logger.warning(f"Failed to save interaction to memory: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID.startswith("<"):
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."

        chunks = []
        for item in results:
            text = item.get("content", {}).get("text", "").strip()
            if text:
                chunks.append(text)

        if not chunks:
            return "No relevant text content found in the knowledge base."

        return "\n---\n".join(chunks)
    except Exception as e:
        logger.warning(f"Knowledge Base retrieval failed: {e}")
        return f"Error retrieving from knowledge base: {str(e)}"


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

loyalty_points = int({loyalty_points})
tier = {repr(tier)}
order_total = float({order_total})
product_category = {repr(product_category)}

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

# 100 points = \$1.00. Floor redeemable points to nearest 500, cap discount at 50% of order_total
max_points_by_order = int((order_total * 0.50) * 100)
usable_points = min(loyalty_points, max_points_by_order)
points_redeemed = (usable_points // 500) * 500
points_discount = points_redeemed / 100.0

subtotal_after_points = max(0.0, order_total - points_discount)
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = round(subtotal_after_points * tier_rate, 2)
final_total = round(subtotal_after_points - tier_discount, 2)
total_savings = round(points_discount + tier_discount, 2)

earn_rate = earn_rates.get(product_category.lower(), 1)
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier_discount_pct": int(tier_rate * 100),
    "tier_discount": tier_discount,
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points
}}
print(json.dumps(result))
"""

    try:
        with code_session(REGION) as client:
            response = client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )
            for event in response.get("stream", []):
                if "result" in event:
                    return json.dumps(event["result"])
            return json.dumps(response)

    except Exception as e:
        logger.warning(f"Code Interpreter failed, using fallback calculation: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        earn_rates = {"standard": 1, "device": 2, "fresh": 5}
        tier_rate = tier_rates.get(tier, 0.0)
        tier_discount = round(order_total * tier_rate, 2)
        final_total = round(order_total - tier_discount, 2)
        earn_rate = earn_rates.get(product_category.lower(), 1)
        points_earned = int(final_total * earn_rate)
        fallback_result = {
            "points_redeemed": 0,
            "points_discount": 0.0,
            "tier_discount_pct": int(tier_rate * 100),
            "tier_discount": tier_discount,
            "final_total": final_total,
            "total_savings": tier_discount,
            "points_earned": points_earned,
            "remaining_points": loyalty_points + points_earned,
            "note": "Computed via tier-only fallback (Code Interpreter unavailable)",
        }
        return json.dumps(fallback_result)


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent, helpful customer support assistant for an e-commerce platform.
You have access to tools for:
1. Order tracking and customer lookup (via Gateway MCP tools)
2. Refund initiation, refund status checks, and return shipping labels (via Gateway MCP tools)
3. Searching the Knowledge Base (`search_knowledge_base`) for product details, return policies, warranties, and loyalty tier benefits
4. Calculating exact loyalty discounts (`calculate_loyalty_discount`) using the Code Interpreter
5. Browsing live web pages (`browser`) when asked to visit a URL or retrieve live web information

Always use the appropriate tool to ground your answers, respect customer preferences stored in context, and provide clear, accurate responses.

Before initiating a refund, look up the order with the order tool and pass its total as the refund amount. Never send $0."""


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "CUST-DEFAULT")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(region=REGION)

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with mcp_client:
            try:
                gateway_tools = mcp_client.list_tools_sync()
                tools.extend(gateway_tools)
                logger.info("Gateway connected successfully.")
            except TimeoutError:
                logger.exception("Gateway tool loading timed out")
                return "I could not reach the order and refund tools. Please try again."
            except ConnectionError:
                logger.exception("Gateway connection failed")
                return "I could not reach the order and refund tools. Please try again."
            except Exception:
                logger.exception("Gateway tool loading failed")
                return "I could not reach the order and refund tools. Please try again."

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )

            response = agent(user_input)
            return response.message["content"][0]["text"]

    except Exception as e:
        logger.error(f"Error during agent invocation: {e}", exc_info=True)
        return f"An error occurred while processing your request: {str(e)}"


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()