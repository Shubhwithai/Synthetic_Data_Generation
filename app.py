import streamlit as st
import json
import csv
from io import StringIO
from typing import List
from pydantic import BaseModel, Field
from portkey_ai import Portkey
import pandas as pd

# Pydantic models
class Conversation(BaseModel):
    user: str = Field(..., description="The message content from the user")
    bot: str = Field(..., description="The message content from the bot")


class TrainingData(BaseModel):
    conversations: List[Conversation] = Field(
        ..., description="A list of conversations between user and bot"
    )


# --- Streamlit App ---
# Main header with hyperlink
st.markdown(
    """
    <h1 style='text-align: center;'>
        <a href='https://portkey.ai/' style='text-decoration: none; color: inherit;'>
            Synthetic Data Generation with Portkey
        </a>
    </h1>
    """,
    unsafe_allow_html=True,
)

# Subheader
st.markdown(
    "<h4 style='text-align: center;'>Generate synthetic datasets with 200+ LLMs</h4>",
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.title("Settings")

# Portkey API Key Input with Link
st.sidebar.markdown(
    """
    <a href='https://app.portkey.ai/api-keys' target='_blank'>
        Fetch Portkey API Key
    </a>
    """,
    unsafe_allow_html=True,
)
portkey_api_key = st.sidebar.text_input("Enter your Portkey API Key:", type="password")

# Model Selection
model = st.sidebar.selectbox(
    "Select Model",
    ["accounts/fireworks/models/llama-v3p1-405b-instruct", "claude-3-5-sonnet-20240620","gpt-4o"],
)

# Virtual Keys (Replace with your actual virtual keys)
virtual_keys = {
    "accounts/fireworks/models/llama-v3p1-405b-instruct": st.secrets[
        "FIREWORKS_VIRTUAL_KEY"
    ],
    "claude-3-5-sonnet-20240620": st.secrets["ANTHROPIC_VIRTUAL_KEY"],
    "gpt-4o": st.secrets["OPENAI_VIRTUAL_KEY"],

}

# Initialize Portkey (only if API key is provided)
fireworks = None
if portkey_api_key:
    try:
        fireworks = Portkey(
            api_key=portkey_api_key,
            virtual_key=virtual_keys.get(model),  # Use get() to handle potential missing keys
        )
    except Exception as e:
        st.sidebar.error(f"Error initializing Portkey: {str(e)}")

# --- Main Content ---

if fireworks:
    # User inputs
    topic = st.text_input("Topic", "General Conversation")
    num_examples = st.number_input(
        "Number of examples to generate", min_value=1, max_value=20, value=5
    )
    custom_instructions = st.text_area("Custom Instructions (Optional)", "")

    # General prompt template
    general_prompt = f"""
    Create {num_examples} examples for instruct dataset for training chatbots to be more anthropomorphic on the topic of {topic}.

    Following the below instructions:

    1. **Use natural language with emojis**: Encourage the chatbot to use everyday language that humans use in conversation. Avoid using overly formal or technical language.

    2. **Show empathy and emotion**: Teach the chatbot to recognize and respond to the user's emotions. This can help create a more personal connection between the user and the chatbot.

    3. **Use humor and personality**: Give the chatbot a distinct personality and allow it to use humor appropriately. This can make interactions more enjoyable, short and engaging.

    4. **Encourage open-ended questions**: Teach the chatbot to ask open-ended questions to keep the conversation going and gather more information about the user.

    DONT EVER MENTION THAT YOU ARE A BOT/ASSISTANT IN THE RESPONSES. ACT LIKE YOU ARE A HUMAN

    {custom_instructions}  # Include custom instructions if provided
    """

    if st.button("Generate Synthetic Data"):
        final_prompt = general_prompt
        with st.spinner("Generating synthetic data..."):
            try:
                completion = fireworks.chat.completions.create(
                    model=model,
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
                            """,
                        },
                        {"role": "user", "content": final_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1000,
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

                    # Create a list of dictionaries for the table
                    table_data = []
                    for i, conv in enumerate(training_data.conversations, 1):
                        table_data.append({"User": conv.user, "Assistant": conv.bot})

                    # Display the data in a table
                    df = pd.DataFrame(table_data)
                    st.table(df)

                    # Prepare CSV data
                    csv_data = StringIO()
                    df.to_csv(csv_data, index=False)

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name="synthetic_data.json",
                            mime="application/json",
                        )
                    with col2:
                        st.download_button(
                            label="Download CSV",
                            data=csv_data.getvalue(),
                            file_name="synthetic_data.csv",
                            mime="text/csv",
                        )

                    # Link to view stats
                    st.markdown(
                        """
                        <a href='https://app.portkey.ai/organisation/a570ceb3-35a3-4a6e-ba3a-ec324c841312/analytics' target='_blank'>
                            View the stats for data generation
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Error parsing the generated data: {str(e)}")
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
else:
    st.warning("Please enter your Portkey API Key in the sidebar to continue.")
