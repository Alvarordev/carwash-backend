from __future__ import annotations

import httpx

GRAPH_API_VERSION = "v19.0"
GRAPH_API_BASE = "https://graph.facebook.com"


async def send_text_message(
    phone_number_id: str,
    access_token: str,
    to: str,
    body: str,
) -> dict:
    """Send a free-form text message via the Meta Cloud API.

    Returns the parsed JSON response from Meta.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    url = f"{GRAPH_API_BASE}/{GRAPH_API_VERSION}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
