# aem_server.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import aiohttp
from urllib.parse import urljoin

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Tool, TextContent
import mcp.types as types
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aem-mcp-server")


class AEMServer:
    def __init__(self, base_url: str, username: str, password: str):
        """Initialize AEM server connection.

        Args:
            base_url: AEM author instance URL (e.g., http://localhost:4502)
            username: AEM username
            password: AEM password
        """
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.auth = aiohttp.BasicAuth(username, password)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(auth=self.auth)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to AEM.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters

        Returns:
            Response data as dictionary
        """
        if not self.session:
            raise RuntimeError("Session not initialized")

        url = urljoin(self.base_url, endpoint)

        try:
            async with self.session.request(method, url, **kwargs) as response:
                response.raise_for_status()

                # Handle different content types
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    return await response.json()
                else:
                    text_content = await response.text()
                    return {"content": text_content, "status": response.status}

        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"AEM request failed: {str(e)}")

    async def get_page_content(self, path: str) -> Dict[str, Any]:
        """Get page content from AEM.

        Args:
            path: Page path (e.g., /content/mysite/en/home)

        Returns:
            Page content data
        """
        endpoint = f"{path}.infinity.json"
        return await self._make_request("GET", endpoint)

    async def create_page(self, parent_path: str, title: str, template: str, name: str = None) -> Dict[str, Any]:
        """Create a new page in AEM.

        Args:
            parent_path: Parent page path
            title: Page title
            template: Page template path
            name: Page name (optional, will be derived from title if not provided)

        Returns:
            Creation response
        """
        if not name:
            name = title.lower().replace(' ', '-').replace('_', '-')

        data = {
            'cmd': 'createPage',
            'parentPath': parent_path,
            'title': title,
            'label': name,
            'template': template
        }

        return await self._make_request("POST", "/bin/wcmcommand", data=data)

    async def update_page_properties(self, path: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Update page properties.

        Args:
            path: Page path
            properties: Properties to update

        Returns:
            Update response
        """
        endpoint = f"{path}/jcr:content"
        return await self._make_request("POST", endpoint, data=properties)

    async def search_content(self, query: str, path: str = "/content") -> Dict[str, Any]:
        """Search for content in AEM.

        Args:
            query: Search query
            path: Search path (default: /content)

        Returns:
            Search results
        """
        params = {
            'query': query,
            'path': path,
            'type': 'cq:Page'
        }

        return await self._make_request("GET", "/bin/querybuilder.json", params=params)

    async def get_asset_info(self, path: str) -> Dict[str, Any]:
        """Get asset information.

        Args:
            path: Asset path

        Returns:
            Asset information
        """
        endpoint = f"{path}.json"
        return await self._make_request("GET", endpoint)

    async def list_children(self, path: str) -> Dict[str, Any]:
        """List child pages/nodes.

        Args:
            path: Parent path

        Returns:
            List of children
        """
        endpoint = f"{path}.1.json"
        return await self._make_request("GET", endpoint)

    async def activate_page(self, path: str) -> Dict[str, Any]:
        """Activate (publish) a page.

        Args:
            path: Page path to activate

        Returns:
            Activation response
        """
        data = {
            'cmd': 'Activate',
            'path': path
        }

        return await self._make_request("POST", "/bin/replicate.json", data=data)

    async def deactivate_page(self, path: str) -> Dict[str, Any]:
        """Deactivate (unpublish) a page.

        Args:
            path: Page path to deactivate

        Returns:
            Deactivation response
        """
        data = {
            'cmd': 'Deactivate',
            'path': path
        }

        return await self._make_request("POST", "/bin/replicate.json", data=data)


# Initialize the MCP server
server = Server("aem-mcp-server")

# Global AEM client instance
aem_client: Optional[AEMServer] = None


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available AEM tools."""
    return [
        Tool(
            name="get_page_content",
            description="Get content and properties of an AEM page",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The page path (e.g., /content/mysite/en/home)"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="create_page",
            description="Create a new page in AEM",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_path": {
                        "type": "string",
                        "description": "Parent page path where the new page will be created"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the new page"
                    },
                    "template": {
                        "type": "string",
                        "description": "Template path for the new page"
                    },
                    "name": {
                        "type": "string",
                        "description": "Page name (optional, derived from title if not provided)"
                    }
                },
                "required": ["parent_path", "title", "template"]
            }
        ),
        Tool(
            name="update_page_properties",
            description="Update properties of an AEM page",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The page path to update"
                    },
                    "properties": {
                        "type": "object",
                        "description": "Properties to update as key-value pairs"
                    }
                },
                "required": ["path", "properties"]
            }
        ),
        Tool(
            name="search_content",
            description="Search for content in AEM",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    },
                    "path": {
                        "type": "string",
                        "description": "Search path (default: /content)",
                        "default": "/content"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_asset_info",
            description="Get information about an AEM asset",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Asset path (e.g., /content/dam/myasset.jpg)"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="list_children",
            description="List child pages or nodes under a given path",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Parent path to list children from"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="activate_page",
            description="Activate (publish) an AEM page",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Page path to activate"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="deactivate_page",
            description="Deactivate (unpublish) an AEM page",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Page path to deactivate"
                    }
                },
                "required": ["path"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[types.TextContent]:
    """Handle tool calls."""

    if not aem_client:
        return [TextContent(type="text", text="Error: AEM client not initialized")]

    try:
        async with aem_client:
            if name == "get_page_content":
                result = await aem_client.get_page_content(arguments["path"])
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "create_page":
                result = await aem_client.create_page(
                    arguments["parent_path"],
                    arguments["title"],
                    arguments["template"],
                    arguments.get("name")
                )
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "update_page_properties":
                result = await aem_client.update_page_properties(
                    arguments["path"],
                    arguments["properties"]
                )
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "search_content":
                result = await aem_client.search_content(
                    arguments["query"],
                    arguments.get("path", "/content")
                )
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "get_asset_info":
                result = await aem_client.get_asset_info(arguments["path"])
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "list_children":
                result = await aem_client.list_children(arguments["path"])
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "activate_page":
                result = await aem_client.activate_page(arguments["path"])
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "deactivate_page":
                result = await aem_client.deactivate_page(arguments["path"])
                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main function to run the AEM MCP server."""
    global aem_client

    # Initialize AEM client with default values
    # In a production environment, these should come from environment variables
    AEM_BASE_URL = "http://localhost:4502"
    AEM_USERNAME = "admin"
    AEM_PASSWORD = "admin"

    try:
        # Initialize AEM client
        aem_client = AEMServer(AEM_BASE_URL, AEM_USERNAME, AEM_PASSWORD)
        logger.info(f"AEM MCP Server initialized for {AEM_BASE_URL}")

        # Run the MCP server with proper stdio handling
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="aem-mcp-server",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise


# Add list_resources handler
@server.list_resources()
async def handle_list_resources() -> list:
    """List available resources."""
    return []


# Add list_prompts handler
@server.list_prompts()
async def handle_list_prompts() -> list:
    """List available prompts."""
    return []


if __name__ == "__main__":
    asyncio.run(main())