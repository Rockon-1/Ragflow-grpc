#!/usr/bin/env python3
"""
Script to register a user, login, and generate a RAGFlow API token.
This token will be used to set the API_KEY environment variable for Docker.
"""

import requests
import os
import sys
import base64
from grpc_ragflow_server.config import RAGFLOW_BASE_URL

# Configuration
HOST_ADDRESS = RAGFLOW_BASE_URL
EMAIL = "lol1@gmail.com"
PASSWORD = "Y5JKJ2mI3Y+Fgji8dLuxTRuxENxv3Uh16nKyIDLVIqUQVZi1lSr/Um5fO9/8dqDWK4oIDkW4erhIx3TC61NEFnrar6IPhQNoyHrPd5tRB4DbF/G0D+qT9iYPEyB+AkOjyJwZWNnBUUoBjJ+uYW0kxcGjZ3A4zQXL/MYG4b5TuvI/1EyQB6ZSa+anchF86zCej3/WwYDKGxe/vNmIUe+dOzC4hQyvkHpcxhOtKcVJj661VCCdOQVXLnru0RENASP+EZ9jaTG7wW2pEKy31hP2zqlafPdL765jHU00EHjbHz/Hulecw1SsMpNo7a98CKiNjplXjrqRNl2IxUrKMADlXA=="

def register():
    """Register a new user with RAGFlow."""
    url = HOST_ADDRESS + "/v1/user/register"
    name = "user"
    register_data = {"email": EMAIL, "nickname": name, "password": PASSWORD}
    res = requests.post(url=url, json=register_data)
    res = res.json()
    if res.get("code") != 0:
        raise Exception(res.get("message"))


def login():
    """Login to RAGFlow and return authorization token."""
    url = HOST_ADDRESS + "/v1/user/login"
    login_data = {"email": EMAIL, "password": PASSWORD}
    response = requests.post(url=url, json=login_data)
    res = response.json()
    if res.get("code") != 0:
        raise Exception(res.get("message"))
    auth = response.headers["Authorization"]
    return auth


def get_api_key_fixture():
    """Generate API key for RAGFlow after registering and logging in."""
    try:
        register()
        print("User registered successfully!")
    except Exception as e:
        print(f"Registration failed or user already exists: {e}")
    
    try:
        auth = login()
        print("Login successful!")
        
        url = HOST_ADDRESS + "/v1/system/new_token"
        auth_header = {"Authorization": auth}
        response = requests.post(url=url, headers=auth_header)
        res = response.json()
        if res.get("code") != 0:
            raise Exception(res.get("message"))
        
        token = res["data"].get("token")
        print("API token generated successfully!")
        return token
    except Exception as e:
        print(f"Error getting API key: {e}")
        sys.exit(1)


def main():
    """Main function to generate and output the API key."""
    try:
        api_key = get_api_key_fixture()
        print(f"API_KEY={api_key}")
        return api_key
    except Exception as e:
        print(f"Failed to generate API key: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
