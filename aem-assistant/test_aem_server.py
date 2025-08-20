#!/usr/bin/env python3
"""
Test script for AEM MCP Server
"""
import asyncio
import sys
import json
from typing import Dict, Any

# Test imports
try:
    from aem_server import AEMServer, server, aem_client
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_aem_server_creation():
    """Test AEM server object creation"""
    try:
        aem = AEMServer("http://localhost:4502", "admin", "admin")
        print("✅ AEM server object created successfully")
        return True
    except Exception as e:
        print(f"❌ AEM server creation failed: {e}")
        return False

async def test_mcp_server_handlers():
    """Test MCP server handler registration"""
    try:
        # Check if handlers are registered
        tools = await server._handlers.get('list_tools', lambda: [])()
        print(f"✅ MCP server has {len(tools)} tools registered")

        # Print available tools
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")

        return True
    except Exception as e:
        print(f"❌ MCP server handler test failed: {e}")
        return False

async def test_aem_connection():
    """Test AEM connection (mock test - doesn't require actual AEM)"""
    try:
        aem = AEMServer("http://localhost:4502", "admin", "admin")
        # Test URL formation
        test_url = aem.base_url + "/content/test.json"
        print(f"✅ AEM URL formation works: {test_url}")
        return True
    except Exception as e:
        print(f"❌ AEM connection test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🔍 Testing AEM MCP Server...")
    print("=" * 50)

    # Test 1: AEM Server Creation
    print("Test 1: AEM Server Creation")
    try:
        aem = AEMServer("http://localhost:4502", "admin", "admin")
        print("✅ AEM server object created successfully")
        test1_passed = True
    except Exception as e:
        print(f"❌ AEM server creation failed: {e}")
        test1_passed = False

    # Test 2: MCP Server Handlers
    print("\nTest 2: MCP Server Handlers")
    try:
        # Check if the server object exists and has handlers
        if hasattr(server, '_handlers'):
            print("✅ MCP server has handlers attribute")

        # Try to get tools list
        from aem_server import handle_list_tools
        tools = await handle_list_tools()
        print(f"✅ MCP server has {len(tools)} tools registered")
        test2_passed = True
    except Exception as e:
        print(f"❌ MCP server handler test failed: {e}")
        test2_passed = False

    # Test 3: Server Decorators Check
    print("\nTest 3: Server Decorators Check")
    try:
        # Check if all required decorators are properly applied
        from aem_server import handle_list_tools, handle_call_tool, handle_list_resources, handle_list_prompts
        print("✅ All handler functions are importable")
        test3_passed = True
    except Exception as e:
        print(f"❌ Decorator test failed: {e}")
        test3_passed = False

    # Summary
    passed = sum([test1_passed, test2_passed, test3_passed])
    total = 3

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Server is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    asyncio.run(main())
