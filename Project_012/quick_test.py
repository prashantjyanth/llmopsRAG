import yaml
from core.agent_builder import ConfigReactAgent
import time
print("=== Testing ConfigReactAgent with Checkpoints ===")

try:
    # Load config
    with open("configs/agents.yaml", "r") as f:
        configs = yaml.safe_load(f)

    # Test with hotel agent
    hotel_cfg = configs["agents"]["hotel_agent"]
    agent = ConfigReactAgent("hotel_agent", hotel_cfg)

    # Show agent info
    info = agent.get_agent_info()
    print(f"\n📊 Agent Info:")
    print(f"  Name: {info['name']}")
    print(f"  Model: {info['model']}")  
    print(f"  Tools: {info['tools_count']} ({', '.join(info['tools'][:3])}{'...' if len(info['tools']) > 3 else ''})")
    print(f"  Checkpoints: {info['has_checkpoints']}")
    
    # Test conversation with memory
    thread_id = "test_conversation"
    
    print(f"\n🧪 Testing conversation memory (Thread: {thread_id})...")
    
    # First interaction
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("👋 Goodbye!")
            break
        response1 = agent.run(user_input, thread_id)
        print(f"Response 1: {response1}")
    
    # # # Second interaction - should remember context
    # response2 = agent.run("search hotel in seattle", thread_id)
    # print(f"Response 2: {response2}")
    
    # # Test different thread (should not remember)
    # response3 = agent.run("What's my name?", "new_thread")
    # print(f"Response 3 (new thread): {response3}")
    
    # # Test specific query
    # response4 = agent.run("Cancel my booking with booking_id=FL12345", thread_id)
    # print(f"Response 4: {response4}")
    
    # print("\n✅ All tests completed!")
    # print("💡 Run agent.chat_session() for interactive mode")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()