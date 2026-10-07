"""Narrow compatibility guard for MCP 1.30 response/cancellation races."""
from anyio import BrokenResourceError, ClosedResourceError
from mcp import ClientSession, types


class ResilientClientSession(ClientSession):
    async def _handle_response(self, message):
        # The SDK pops the waiter before awaiting send(). Cancellation can close
        # that waiter in between; only that abandoned response should be lost.
        root = message.message.root
        stream = None
        if isinstance(root, (types.JSONRPCResponse, types.JSONRPCError)):
            response_id = self._normalize_request_id(root.id)
            stream = self._response_streams.get(response_id)
        try:
            await super()._handle_response(message)
        except (ClosedResourceError, BrokenResourceError):
            if stream is None or stream.statistics().open_receive_streams != 0:
                raise
