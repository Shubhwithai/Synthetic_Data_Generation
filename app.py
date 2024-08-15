import streamlit as st
import json
import csv
from io import StringIO
from typing import List
from pydantic import BaseModel, Field
from portkey_ai import Portkey

# Pydantic models
class Conversation(BaseModel):
    user: str = Field(..., description="The message content from the user")
    bot: str = Field(..., description="The message content from the bot")

class TrainingData(BaseModel):
    conversations: List[Conversation] = Field(
        ...,
        description="A list of conversations between user and bot"
    )

# Initialize Portkey
@st.cache_resource
def initialize_portkey():
    return Portkey(
        api_key=st.secrets["PORTKEY_API_KEY"],
        virtual_key=st.secrets["FIREWORKS_VIRTUAL_KEY"]
    )

fireworks = initialize_portkey()

# Streamlit app
st.title("Synthetic Data Generation App")

# Display smaller text aligned to the left of the title
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center;'>
        <h1>Synthetic Data Generation App</h1>
        <small style='margin-left: auto;'>Made By BuildFastWithAI</small>
    </div>
    """, 
    unsafe_allow_html=True
)

# User inputs
language = st.selectbox("Select language", ["English", "Hindi", "Marathi", "Hinglish", "Telgue"])
num_examples = st.number_input("Number of examples to generate", min_value=1, max_value=20, value=5)

# General prompt template (hidden by default)
general_prompt = f"""
Create {num_examples} examples for instruct dataset for training chatbots to be more anthropomorphic in {language}.

Following the below instructions:

1. **Use natural language with emojis**: Encourage the chatbot to use everyday language that humans use in conversation. Avoid using overly formal or technical language.

2. **Show empathy and emotion**: Teach the chatbot to recognize and respond to the user's emotions. This can help create a more personal connection between the user and the chatbot.

3. **Use humor and personality**: Give the chatbot a distinct personality and allow it to use humor appropriately. This can make interactions more enjoyable, short and engaging.

4. **Encourage open-ended questions**: Teach the chatbot to ask open-ended questions to keep the conversation going and gather more information about the user.

DONT EVER MENTION THAT YOU ARE A BOT/ASSISTANT IN THE RESPONSES. ACT LIKE YOU ARE A HUMAN
"""

if st.button("Generate Synthetic Data"):
    final_prompt = general_prompt
    with st.spinner("Generating synthetic data..."):
        completion = fireworks.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": """Generate a list of conversations between a user and a bot in the following JSON format:
                    {
                        "conversations": [
                            {
                                "user": "User's message goes here",
                                "bot": "Bot's response goes here"
                            },
                            {
                                "user": "Next user message",
                                "bot": "Next bot response"
                            },
                            // Additional conversations follow the same structure
                        ]
                    }
                    Each conversation should consist of a user message followed by a bot response.
                    The conversations should be coherent and demonstrate a variety of user queries and bot responses.
                    Generate conversation pairs.
                    """
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=0.7,
            max_tokens=8000,
            top_p=1,
            stream=False,
            response_format={"type": "json_object"},
            stop=None,
        )
        
        json_data = completion.choices[0].message.content
        
        # Parse and validate the data
        try:
            parsed_data = json.loads(json_data)
            training_data = TrainingData(**parsed_data)
            
            # Display the generated conversations
            st.subheader("Generated Conversations")
            for i, conv in enumerate(training_data.conversations, 1):
                st.write(f"Conversation {i}:")
                st.write(f"User: {conv.user}")
                st.write(f"Bot: {conv.bot}")
                st.write("---")
            
            # Prepare CSV data
            csv_data = StringIO()
            csv_writer = csv.writer(csv_data)
            csv_writer.writerow(["Conversation", "Role", "Message"])
            for i, conv in enumerate(training_data.conversations, 1):
                csv_writer.writerow([i, "User", conv.user])
                csv_writer.writerow([i, "Bot", conv.bot])
            
            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="synthetic_data.json",
                    mime="application/json"
                )
            with col2:
                st.download_button(
                    label="Download CSV",
                    data=csv_data.getvalue(),
                    file_name="synthetic_data.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error parsing the generated data: {str(e)}")

# # Add some instructions for the user
# st.sidebar.header("Instructions")
# st.sidebar.write("""
# 1. Select the desired language and number of examples to generate.
# 2. Optionally, show and edit the general prompt by checking the box.
# 3. Add any additional custom instructions in the optional field.
# 4. Click the 'Generate Synthetic Data' button to create conversations.
# 5. Review the generated conversations displayed on the page.
# 6. Download the data as JSON or CSV using the provided buttons.
# """)
