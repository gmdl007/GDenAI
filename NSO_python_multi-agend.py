#!/usr/bin/env python
# coding: utf-8

# In[3]:


from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings
from llama_index.core import (
    load_index_from_storage,
    load_indices_from_storage,
    load_graph_from_storage,
)


# In[4]:


import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
token_url = os.getenv('TOKEN_URL')
llm_endpoint = os.getenv('LLM_ENDPOINT')
appkey = os.getenv('APP_KEY')
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')
api_base_url = os.getenv('API_BASE_URL')


# In[5]:


# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

from flask import Flask, request, render_template_string, redirect, url_for
import logging
import sys

# Logging setup
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# In[8]:


from llama_index.llms.azure_openai import AzureOpenAI

import logging
import sys
import json


logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
import base64
import requests

# print(base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8'))
auth_key = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
headers = {
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Authorization": f"Basic {auth_key}",
}

# Make a POST request to retrieve the token
token_response = requests.post(token_url, headers=headers, data="grant_type=client_credentials")
token = token_response.json().get("access_token")

user_param = json.dumps({"appkey": appkey})


llm = AzureOpenAI(azure_endpoint=llm_endpoint,
                  #model= 'gpt-4o-mini',
                  api_version="2024-07-01-preview",
                  deployment_name='gpt-4o-mini',
                  api_key=token,
                  max_tokens=500,
                  temperature=0.1,
                  additional_kwargs={"user": f'{{"appkey": "{appkey}"}}'}
                 )


# In[4]:


llm = None

def initialize_llm(token_url, headers, llm_endpoint, appkey):
    # Retrieve the token via a POST request
    token_response = requests.post(token_url, headers=headers, data="grant_type=client_credentials")
    token = token_response.json().get("access_token")
    
    global llm  # Use the global llm variable
    
    llm = None
    
    # Overwrite the LLM
    llm = AzureOpenAI(
        azure_endpoint=llm_endpoint,
        api_version="2024-07-01-preview",
        deployment_name='gpt-4o-mini',
        api_key=token,
        max_tokens=3000,
        temperature=0.1,
        additional_kwargs={"user": f'{{"appkey": "{appkey}"}}'}
    )
    
    # Set the LLM in Settings
    Settings.llm = llm
    Settings.context_window = 8000



# In[5]:


initialize_llm(token_url, headers, llm_endpoint, appkey)


# In[6]:


#Settings.embed_model=embed_model
#Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
#Settings.num_output = 512
Settings.context_window = 8000
Settings.chunk_size = 1024
Settings.llm = llm


# In[ ]:





# In[7]:


from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool


# In[8]:


def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


# In[14]:


def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)


# In[10]:


agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)


# In[11]:


response = agent.chat("What is 20+(2*4)? Calculate step by step ")


# In[12]:


response = agent.chat("What is 2+2*4")
print(response)


# In[13]:


prompt_dict = agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}")


# In[9]:


import ncs
import ncs.maapi as maapi
import ncs.maagic as maagic
m = maapi.Maapi()
import io
import sys
import re
import os

m.start_user_session('admin','test_context_1')
t = m.start_write_trans()
root = maagic.get_root(t)
#t.finish


# In[10]:


for i in root.devices.device:
    print(i.name)


# In[11]:


def show_all_devices():
    """
    Find out all available routers in the lab, return their names. This is helpful if you dont know what devices or router are out there and you dont know where to start.

    Args:
        None
    
    Returns:
        str: the list of names of routers in the lab or network
    """
    if hasattr(root, 'devices') and hasattr(root.devices, 'device'):
        # Collect router names into a list
        router_names = [device.name for device in root.devices.device]
        
        # Print each router name
        for name in router_names:
            print(name)
        
        # Return concatenated names as a string
        return ', '.join(router_names)
    else:
        print("No devices found.")
        return "No devices found."


# In[12]:


items=m.list_rollbacks(5)
for item in items:
    print(item.label)
    print(item.fixed_nr)
    print(item.creator)


# In[13]:


def roll_back(steps=0):
    """
    Rolls back to a specified commit.

    Args:
        steps (int, optional): The number of steps to roll back. Defaults to 0, 
                               which rolls back to the most recent commit.
                               For example:
                               - roll_back() rolls back 1 step (rollback ID 0).
                               - roll_back(1) rolls back 2 steps.
                               - roll_back(n) rolls back (n + 1) steps.

    Returns:
        None
    """
    import ncs.maapi as m  # Ensure the correct library is imported for transactions

    rollback_id = steps  # Use the input number as rollback ID (0 for the latest commit)
    with m.single_write_trans('admin', 'python') as t:
        t.load_rollback(rollback_id)
        t.apply()


# In[14]:


#roll_back(1)


# In[15]:


def configure_subinterface(device_name, subinterface_id, ip_address, subnet_mask):
    """
    Configures a subinterface with specified parameters on a device or router

    Args:
        device_name (str): The name of the device to configure.
        subinterface_id (str): The subinterface identifier (e.g., '0/0/0/0.200').
        ip_address (str): The IPv4 address to assign to the subinterface.
        subnet_mask (str): The subnet mask for the IP address.

    Returns:
        None
    """
    with ncs.maapi.single_write_trans('admin', 'python') as t:
        root = ncs.maagic.get_root(t)
        device = root.devices.device[device_name]
        device.config.cisco_ios_xr__interface.GigabitEthernet_subinterface.GigabitEthernet.create(subinterface_id)
        subint = device.config.cisco_ios_xr__interface.GigabitEthernet_subinterface.GigabitEthernet[subinterface_id]
        subint.ipv4.address.ip = ip_address
        subint.ipv4.address.mask = subnet_mask
        t.apply()


# In[16]:


#configure_subinterface("xr9kv-1", "0/0/0/0.108", "192.0.3.1", "255.255.255.0")


# In[17]:


def iterate_devices_AND_cmd(cmds):
    """
    Example of how to loop over devices in NSO and execute actions or changes per each device.
    This example iterates over devices and prints the result of the specified commands.
    """
    results = []  # Initialize a list to store the results
    with ncs.maapi.single_write_trans('admin', 'python', groups=['ncsadmin']) as t:
        root = ncs.maagic.get_root(t)
        for box in root.devices.device:
            for cmd in cmds:
                try:
                    # Get the 'show' action object
                    show = box.live_status.__getitem__('exec').any
                    
                    # Prepare the input for the command
                    inp = show.get_input()
                    inp.args = [cmd]
                    
                    # Execute the command and get the result
                    r = show.request(inp)
                    
                    # Format the result and print it
                    show_cmd = 'result of Show Command "{}" for Router Name {}: {}'.format(cmd, box.name, r.result)
                    print(show_cmd)
                    
                    # Append the result to the list
                    results.append(show_cmd)
                    
                except Exception as e:
                    print(f"Failed to execute command '{cmd}' on device {box.name}: {e}")
    
    # Return the list of results after the loop completes
    return results

# Example usage:
commands = ['show version', "show ipv4 int brief"]
results = iterate_devices_AND_cmd(commands)


# In[18]:


def iterate_devices_AND_cmd(cmd):
    """
    Execute a single command on all devices in NSO and print the results.

    Args:
        cmd (str): The command to execute on each device.

    Returns:
        list: A list of strings containing the results of the command execution.
    """
    results = []  # Initialize a list to store the results
    with ncs.maapi.single_write_trans('admin', 'python', groups=['ncsadmin']) as t:
        root = ncs.maagic.get_root(t)
        for box in root.devices.device:
            try:
                # Get the 'show' action object
                show = box.live_status.__getitem__('exec').any
                
                # Prepare the input for the command
                inp = show.get_input()
                inp.args = [cmd]
                
                # Execute the command and get the result
                r = show.request(inp)
                
                # Format the result and print it
                show_cmd = 'Result of Show Command "{}" for Router Name {}: {}'.format(cmd, box.name, r.result)
                print(show_cmd)
                
                # Append the result to the list
                results.append(show_cmd)
                
            except Exception as e:
                print(f"Failed to execute command '{cmd}' on device {box.name}: {e}")
    
    # Return the list of results after the loop completes
    return results

# Example usage:
command = "show version"
results = iterate_devices_AND_cmd(command)


# In[19]:


def iterate(cmds):
    """
    iterate the commands on every router.
    
    Args:
        the cmds are the commands to be executed on every router
    
    Returns:
        str: the output of command of every router.
    """
    return iterate_devices_AND_cmd(cmds)


# In[20]:


iterate('show ipv4 int brief')


# In[21]:


def execute_command_on_router(router_name, command):
    """
    Executes a single command on a specific router using NSO and returns the result.
    
    Args:
        router_name (str): The name of the router to execute the command on.
        command (str): The command to execute.
    
    Returns:
        str: The result of the command execution.
    """
    try:
        # Initialize a write transaction
        with ncs.maapi.single_write_trans('admin', 'python', groups=['ncsadmin']) as t:
            root = ncs.maagic.get_root(t)
            
            # Locate the specific device
            device = root.devices.device[router_name]
            
            # Get the 'show' action object
            show = device.live_status.__getitem__('exec').any
            
            # Prepare the input for the command
            inp = show.get_input()
            inp.args = [command]
            
            # Execute the command and get the result
            r = show.request(inp)
            
            # Format the result and return
            result = f'Result of Show Command "{command}" for Router "{router_name}": {r.result}'
            print(result)
            return result
            
    except KeyError:
        error_msg = f"Device '{router_name}' not found in NSO."
        print(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Failed to execute command '{command}' on device '{router_name}': {e}"
        print(error_msg)
        return error_msg


# In[22]:


def show_run(router_name, arg):
    """
    Retrieves the router version using the 'show version' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The version information of the router.
    """
    command = "show run "
    return execute_command_on_router(router_name, command)



def get_router_version(router_name):
    """
    Retrieves the router version using the 'show version' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The version information of the router.
    """
    command = "show version"
    return execute_command_on_router(router_name, command)

def get_router_Lo0_IP(router_name):
    """
    Retrieves the router Loopback0 IP address using the 'show ip interface loopback0' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The version information of the router.
    """
    command = "show ip interface loopback0"
    return execute_command_on_router(router_name, command)

def get_router_clock(router_name):
    """
    Retrieves the router current time using the 'show clock' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The version information of the router.
    """
    command = "show clock"
    return execute_command_on_router(router_name, command)


def show_router_interfaces(router_name):
    """
    Retrieves the summary of router interface status using the 'show ipv4 interface brief' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The interface status information of the router.
    """
    command = "show ipv4 interface brief"
    return execute_command_on_router(router_name, command)

def get_router_ip_routes(router_name, prefix):
    """
    Retrieves a particular IPv4 route using the 'show route ipv4 <prefix>' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
        prefix (str): The IP prefix (e.g., "192.168.1.0/24") to look up in the routing table.
    
    Returns:
        str: The routing information for the specified prefix.
    """
    command = f"show route ipv4 {prefix}"  # Correctly inject the prefix into the command string
    return execute_command_on_router(router_name, command)


def get_router_bgp_summary(router_name):
    """
    Retrieves the BGP summary information using the 'show bgp ipv4 unicast summary' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The BGP summary information of the router.
    """
    command = "show bgp ipv4 unicast summary"
    return execute_command_on_router(router_name, command)

def get_router_isis_neighbors(router_name):
    """
    Retrieves the ISIS neighbors information using the 'show isis neighbors' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The ISIS neighbors information of the router.
    """
    command = "show isis neighbors"
    return execute_command_on_router(router_name, command)

def get_router_ospf_summary(router_name):
    """
    Retrieves the OSPF summary information using the 'show ospf vrf all-inclusive summary' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The OSPF summary information of the router.
    """
    command = "show ospf vrf all-inclusive summary"
    return execute_command_on_router(router_name, command)

def get_router_ospf_neigh(router_name):
    """
    Retrieves the OSPF neighbor information using the 'show ospf neighbor' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The OSPF neighbor information of the router.
    """
    command = "show ospf neighbor"
    return execute_command_on_router(router_name, command)

def get_router_control_plane_cpu(router_name):
    """
    Retrieves the router control plane CPU usage using the 'show processes cpu' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The control plane CPU usage information of the router.
    """
    command = "show processes cpu"
    return execute_command_on_router(router_name, command)

def get_router_memory_usage(router_name):
    """
    Retrieves the router memory usage using the 'show processes memory' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The memory usage information of the router.
    """
    command = "show processes memory"
    return execute_command_on_router(router_name, command)

def ping_router(router_name, ip_address):
    """
    Pings a ip address using the 'ping' command on a router, return the result of the ping command
    
    Args:
        router_name (str): The name of the router to execute the command on.
        ip_address (str): The IP address to ping.
    
    Returns:
        str: The result of the ping command.
    """
    command = f"ping {ip_address} source Loopback 0"
    return execute_command_on_router(router_name, command)

def traceroute_router(router_name, ip_address):
    """
    Performs a traceroute to a device using the 'traceroute' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
        ip_address (str): The IP address to traceroute.
    
    Returns:
        str: The result of the traceroute command.
    """
    command = f"traceroute {ip_address} source Loopback 0"
    return execute_command_on_router(router_name, command)

def lldp_nei(router_name):
    """
    find the connected neighbors with 'show lldp neighbor' command.
    
    Args:
        router_name (str): The name of the router to execute the command on.
    
    Returns:
        str: The result of the 'show lldp neighbor' command.
    """
    command = "show lldp neighbor"
    return execute_command_on_router(router_name, command)

def mpls_lfib(router_name, prefix=None):
    """
    Check the MPLS Label Forwarding Information Base (LFIB).

    If no prefix is provided, it executes the 'show mpls forwarding' command
    to display the complete LFIB. If a prefix is provided, it executes
    'show mpls forwarding prefix <prefix>' to display specific MPLS LFIB
    information for the given prefix.

    Args:
        router_name (str): The name of the router to execute the command on.
        prefix (str, optional): The specific prefix to filter MPLS LFIB information. 
                                Defaults to None.

    Returns:
        str: The result of the MPLS LFIB command execution.
    """
    if prefix:
        command = f"show mpls forwarding prefix {prefix}"
    else:
        command = "show mpls forwarding"
    
    return execute_command_on_router(router_name, command)


# In[23]:


def get_router_logs(router_name, match_string=None):
    """
    Retrieves router logs using the 'show logging last 50' command or filters by the specified string if provided.
    
    Args:
        router_name (str): The name of the router to execute the command on.
        match_string (str, optional): The string to match within the logs. If None, retrieves the last 50 logs.
    
    Returns:
        str: The filtered logs or the last 50 logs of the router, depending on whether a match_string is provided.
    """
    if match_string:
        # If a match string is provided, retrieve the logs with string matching
        full_logs = execute_command_on_router(router_name, f"show logging | include {match_string}")
        
        if full_logs:
            result = f"Logs matching '{match_string}' on router '{router_name}':\n{full_logs}"
        else:
            result = f"No logs matching '{match_string}' found on router '{router_name}'."
    else:
        # If no match string is provided, retrieve the last 50 logs
        command = "show logging last 50"
        full_logs = execute_command_on_router(router_name, command)
        
        result = f"Last 50 logs on router '{router_name}':\n{full_logs}"

    return result


# In[24]:


import requests
from requests.auth import HTTPBasicAuth

def fetch_nso_config(device_name, config_path):
    """
    Get the configuration of config path for a specific networking device

    Args:
        device_name (str): Name of the networking device
        config_path (str): the config path.

    Returns:
        dict: Parsed JSON response from the API.
        None: If an error occurs.
    """
    # Define base URL and credentials
    base_url = api_base_url
    username = username
    password = password

    # Construct full API URL
    url = f"{base_url}={device_name}/config/tailf-ned-cisco-ios-xr:{config_path}"

    # Set headers
    headers = {
        "Accept": "application/yang-data+json",
    }

    try:
        # Make the GET request
        response = requests.get(url, headers=headers, auth=HTTPBasicAuth(username, password))

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


# In[25]:


# Creating the FunctionTool for different operational commands
check_version_tool = FunctionTool.from_defaults(fn=get_router_version)
check_time_tool = FunctionTool.from_defaults(fn=get_router_clock)
check_Lo0_tool = FunctionTool.from_defaults(fn=get_router_Lo0_IP)
check_interfaces_tool = FunctionTool.from_defaults(fn=show_router_interfaces)
check_ip_route_tool = FunctionTool.from_defaults(fn=get_router_ip_routes)
check_bgp_summary_tool = FunctionTool.from_defaults(fn=get_router_bgp_summary)
check_isis_neighbors_tool = FunctionTool.from_defaults(fn=get_router_isis_neighbors)
check_ospf_summary_tool = FunctionTool.from_defaults(fn=get_router_ospf_summary)

check_ospf_neigh_tool = FunctionTool.from_defaults(fn=get_router_ospf_neigh)

check_cpu_usage_tool = FunctionTool.from_defaults(fn=get_router_control_plane_cpu)
check_memory_usage_tool = FunctionTool.from_defaults(fn=get_router_memory_usage)

# Create tools for ping and traceroute
ping_tool = FunctionTool.from_defaults(fn=ping_router)
traceroute_tool = FunctionTool.from_defaults(fn=traceroute_router)

logging_tool = FunctionTool.from_defaults(fn=get_router_logs)
iterate_cmd_tool = FunctionTool.from_defaults(fn=iterate)
lldp_neigh_tool = FunctionTool.from_defaults(fn=lldp_nei)
create_sub_int_tool = FunctionTool.from_defaults(fn=configure_subinterface)
rollback_tool = FunctionTool.from_defaults(fn=roll_back)
mpls_lfib_tool = FunctionTool.from_defaults(fn=mpls_lfib)
all_router_tool = FunctionTool.from_defaults(fn=show_all_devices)
fetch_config_tool = FunctionTool.from_defaults(fn=fetch_nso_config)


# In[26]:


config = fetch_nso_config("xr9kv-1", "router/")
print(config)


# In[ ]:





# In[ ]:





# In[27]:


List_Tools = [
        check_version_tool, 
        check_time_tool, 
        check_Lo0_tool, 
        ping_tool,
        traceroute_tool,
        logging_tool,
        check_interfaces_tool, 
        check_ip_route_tool, 
        check_bgp_summary_tool, 
        check_isis_neighbors_tool, 
        check_ospf_summary_tool,
        check_ospf_neigh_tool,
        check_cpu_usage_tool, 
        check_memory_usage_tool,
        iterate_cmd_tool,
        lldp_neigh_tool,
        create_sub_int_tool,
        rollback_tool,
        mpls_lfib_tool,
        all_router_tool,
        fetch_config_tool
]


# In[33]:


agent = ReActAgent.from_tools(List_Tools, llm=llm, verbose=True, max_iterations=10000)


# In[34]:


def kick_agent():
    global agent  # Ensure we're modifying the global 'agent' variable
    agent = None  # Clear the agent
    agent = ReActAgent.from_tools(List_Tools, llm=llm, verbose=True, max_iterations=10000)
    return agent  # Optional, though 'agent' is modified globally


# In[35]:


def initialize_agent():
    global agent, llm
    # Reinitialize the LLM
    llm = None
    initialize_llm(token_url, headers, llm_endpoint, appkey)  # Ensure these variables are defined globally or passed here
    
    # Initialize the agent
    kick_agent()


# In[36]:


# Flask app initialization
app = Flask(__name__)

# HTML template
form_template = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Query Interface</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
    }
    h1 {
      font-size: 24px;
      color: #333;
    }
    form {
      margin-bottom: 20px;
    }
    textarea {
      width: 100%;
      height: 50px;
      padding: 10px;
      font-size: 16px;
      border: 1px solid #ccc;
      border-radius: 4px;
      resize: none;
    }
    input[type="submit"] {
      padding: 10px 20px;
      background-color: #4CAF50;
      color: white;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 16px;
    }
    input[type="submit"]:hover {
      background-color: #45a049;
    }
    pre {
      background-color: #f4f4f4;
      padding: 15px;
      border-radius: 4px;
      white-space: pre-wrap;
      word-wrap: break-word;
      font-family: 'Courier New', Courier, monospace;
      font-size: 14px;
      color: #333;
    }
  </style>
</head>
<body>
  <h1>Query the Agent</h1>
  <form action="/" method="post">
    <textarea name="text" placeholder="Enter your query here" required></textarea>
    <br><br>
    <input type="submit" value="Submit">
  </form>
  {% if response %}
    <h2>Response:</h2>
    <pre>{{ response }}</pre>
  {% endif %}
  <form action="/reset-agent" method="post">
    <button type="submit" style="background-color: #ff6347; color: white; border: none; padding: 10px; border-radius: 4px; cursor: pointer;">Reset Agent</button>
  </form>
</body>
</html>
"""

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    if request.method == "POST":
        query_text = request.form.get("text", "")
        if query_text:
            # Query the agent
            response = agent.chat(query_text)
    return render_template_string(form_template, response=response)


# Reset agent route
@app.route("/reset-agent", methods=["POST"])
def reset_agent():
    initialize_agent()  # Reinitializes both LLM and Agent
    logging.info("Agent and LLM have been reset.")
    return redirect(url_for("home"))


if __name__ == "__main__":
    # Run the app with SSL
    app.run(host="0.0.0.0", port=5602, ssl_context=('./myproject_2/certs/cert.pem', './myproject_2/certs/key.pem'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




