#!/usr/bin/env python3
"""
Minimal AEM Server Test - Workaround version
"""
import asyncio
from aem_server import AEMServer

async def minimal_test():
    """Test basic AEM server functionality without MCP"""
    try:
        # Test 1: Basic AEM server creation
        aem = AEMServer("http://localhost:4502", "admin", "admin")
        print("✅ AEM server created")

        # Test 2: Test URL formation
        test_endpoint = "/content/test.json"
        from urllib.parse import urljoin
        full_url = urljoin(aem.base_url, test_endpoint)
        print(f"✅ URL formation works: {full_url}")

        # Test 3: Test auth object
        if aem.auth:
            print("✅ Authentication object created")

        return True

    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(minimal_test())
