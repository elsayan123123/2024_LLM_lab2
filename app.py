import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("xiaojingyan/lora_model_r16_merged16")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response


def chat_interface():
    with gr.Blocks(css="""
        #send_button {
            background-color: grey;
            color: white;
            border: none;
            padding: 8px 16px;
            font-size: 16px;
            border-radius: 4px;
            cursor: not-allowed;
        }
        #send_button.active {
            background-color: blue;
            cursor: pointer;
        }
    """) as demo:
        gr.Markdown(
            """
            ## 🤖 Chatbot Interface
            Welcome to the enhanced chatbot interface! Customize settings below and interact with the bot in the chat window.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                chat = gr.Chatbot()  # Default Chatbot component for user and assistant
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Your Message",
                    lines=1,
                    interactive=True,
                )
                submit = gr.Button("Send", elem_id="send_button")
                typing_indicator = gr.Markdown("")  # Placeholder for typing indicator

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                system_message = gr.Textbox(
                    value="You are a friendly chatbot.",
                    label="System Message",
                    lines=3,
                )
                max_tokens = gr.Slider(
                    minimum=1, maximum=2048, value=512, step=1, label="Max New Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p"
                )
                reset_button = gr.Button("Reset Chat")  # Reset button to clear history

        history = gr.State([])  # Chat history state

        # Define interaction logic
        def user_input(
            user_message, chat_history, system_msg, max_t, temp, top_p_val
        ):
            if user_message:
                chat_history.append((user_message, None))  # Add user message
                yield chat_history, "", "Assistant is typing..."
                response = respond(
                    user_message, chat_history, system_msg, max_t, temp, top_p_val
                )
                for partial_response in response:
                    chat_history[-1] = (user_message, partial_response)  # Update assistant response
                    yield chat_history, "", "Assistant is typing..."
                yield chat_history, "", ""

        submit.click(
            user_input,
            inputs=[msg, history, system_message, max_tokens, temperature, top_p],
            outputs=[chat, msg, typing_indicator],
            show_progress=True,
        )

        msg.submit(
            user_input,
            inputs=[msg, history, system_message, max_tokens, temperature, top_p],
            outputs=[chat, msg, typing_indicator],
            show_progress=True,
        )

        # Change button class dynamically
        def toggle_button_color(text):
            if text.strip():
                return gr.update(elem_classes=["active"])
            else:
                return gr.update(elem_classes=[])

        msg.change(toggle_button_color, inputs=msg, outputs=submit)

        # Reset chat
        def reset_chat():
            return [], "", "", []  # Clear chat, message, typing indicator, and history state

        reset_button.click(reset_chat, inputs=[], outputs=[chat, msg, typing_indicator, history])

    return demo


if __name__ == "__main__":
    demo = chat_interface()
    demo.launch()
