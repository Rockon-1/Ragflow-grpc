"""
RAGFlow gRPC Server Application

Main entry point for the RAGFlow gRPC server.
This module starts the gRPC server or client
"""
import asyncio
import sys
import os
import logging
from typing import Optional
from get_ragflow_token import main as get_ragflow_token_main

from dotenv import load_dotenv


# Add the grpc_ragflow_server directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'grpc_ragflow_server'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def start_server():
    """Start the RAGFlow gRPC server"""
    try:
        from grpc_ragflow_server.server import serve
        logger.info("Starting RAGFlow gRPC server...")
        load_dotenv(override=True) 
        API_KEY = os.environ.get("API_KEY", "None")
        logger.info(f"Using API_KEY: {API_KEY}")
        if API_KEY == None or API_KEY == "None":
            logger.warning("API_KEY is not set. ")
            get_ragflow_token_main()
        await serve()
    except ImportError as e:
        logger.error(f"Failed to import server module: {e}")
        logger.error("Make sure protobuf files are generated. Run: make protobuf")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


async def run_client_examples():
    """Run client examples"""
    try:
        from grpc_client_example import main as client_main
        logger.info("Running client examples...")
        await client_main()
    except ImportError as e:
        logger.error(f"Failed to import client example: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to run client examples: {e}")
        sys.exit(1)


async def main(mode: Optional[str] = None):
    """
    Main function to run the RAGFlow gRPC application
    
    Args:
        mode: 'server', 'client', or None (interactive choice)
    """
    logger.info("RAGFlow gRPC Application")
    
    if mode is None:
        print("\nChoose mode:")
        print("1. Start gRPC Server")
        print("2. Run Client Examples")
      
        
        try:
            choice = input("\nEnter your choice (1-2): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            return
        
        if choice == '1':
            mode = 'server'
        elif choice == '2':
            mode = 'client'
  
        else:
            print("Invalid choice. Exiting...")
            return
    
    try:
        if mode == 'server':
            await start_server()
        elif mode == 'client':
            await run_client_examples()

        else:
            print(f"Unknown mode: {mode}")
            return
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


def cli():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAGFlow gRPC Application')
    parser.add_argument(
        'mode', 
        choices=['server', 'client', ], 
        nargs='?',
        help='Mode to run: server or client'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run the application
    asyncio.run(main(args.mode))


if __name__ == "__main__":
    # Check if running with command line arguments
    if len(sys.argv) > 1:
        cli()
    else:
        # Run interactively
        asyncio.run(main())
