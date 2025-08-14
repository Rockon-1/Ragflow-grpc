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
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

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


def export_api_key_to_env(api_key: Optional[str] = None, 
                          env_file: str = ".env", 
                          key_name: str = "API_KEY",
                          overwrite: bool = True) -> str:
    """
    Export an API key to a .env file.
    
    Args:
        api_key: The API key to export. If None, a new one will be generated.
        env_file: Path to the .env file (default: ".env")
        key_name: The name of the environment variable to set (default: "RAGFLOW_API_KEY")
        overwrite: Whether to overwrite the key if it already exists in the file
        
    Returns:
        The API key that was exported
    """
    # Generate a new API key if one wasn't provided
    if api_key is None:
        api_key = "API_KEY"
    
    env_path = Path(env_file)
    existing_content = {}
    new_content = []
    key_exists = False
    
    # Read existing content if the file exists
    if env_path.exists():
        with open(env_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    new_content.append(line)
                    continue
                    
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    existing_content[key] = value.strip()
                    
                    if key == key_name:
                        key_exists = True
                        if not overwrite:
                            # Keep existing value
                            new_content.append(line)
                        else:
                            # Will be added later
                            pass
                    else:
                        new_content.append(line)
    
    # Add or update the API key
    if not key_exists or overwrite:
        new_content.append(f"{key_name}={api_key}")
    
    # Write the updated content back to the file
    with open(env_path, "w") as f:
        f.write("\n".join(new_content))
        if new_content and not new_content[-1].endswith("\n"):
            f.write("\n")
    
    print(f"API key {'updated' if key_exists else 'added'} in {env_file}")
    return api_key


def main():
    """Main function to generate and output the API key."""
    try:
        api_key = get_api_key_fixture()
        print(f"API_KEY={api_key}")

        export_api_key_to_env(api_key,key_name="API_KEY")
        load_dotenv(override=True) 
        print("API value in .env is ",os.environ.get("API_KEY", "75bb8d62790a11f0bcd2a26a1b104320"))
        return api_key
    except Exception as e:
        print(f"Failed to generate API key: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
