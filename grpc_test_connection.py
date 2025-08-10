"""
Test gRPC Connection

This script tests the connection to the RAGFlow gRPC server.
"""
import asyncio
import grpc
import logging
from grpc_ragflow_server.config import GRPC_HOST, GRPC_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_grpc_connection():
    """Test if gRPC server is reachable"""
    try:
        # Create channel with timeout
        channel = grpc.aio.insecure_channel(
            f'{GRPC_HOST}:{GRPC_PORT}',
            options=[
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 5000),
                ('grpc.keepalive_permit_without_calls', True),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 300000)
            ]
        )
        
        # Wait for channel to be ready with timeout
        logger.info(f"Attempting to connect to gRPC server at {GRPC_HOST}:{GRPC_PORT}")
        
        await asyncio.wait_for(
            channel.channel_ready(), 
            timeout=10.0
        )
        
        logger.info(" Successfully connected to gRPC server!")
        
        # Close the channel
        await channel.close()
        return True
        
    except asyncio.TimeoutError:
        logger.error("failed- Connection timeout - gRPC server is not responding")
        return False
    except grpc.RpcError as e:
        logger.error(f"failed- gRPC error: {e}")
        return False
    except Exception as e:
        logger.error(f"failed- Connection failed: {e}")
        return False


async def main():
    """Main function"""
    logger.info("Testing gRPC connection...")
    
    success = await test_grpc_connection()
    
    if success:
        logger.info("\n Connection test passed! You can now run the gRPC client examples.")
        logger.info("Try running:")
        logger.info("  uv run python simple_grpc_client.py")
        logger.info("  uv run python grpc_client_example.py")
    else:
        logger.info("\n Connection test failed!")
        logger.info("Make sure the gRPC server is running:")
        logger.info("  uv run python grpc_ragflow_server/server.py")


if __name__ == "__main__":
    asyncio.run(main())
