import gradio as gr
from .data.app.qa_chain import answer_question

def qa_interface(question):
    if not question or question.strip() == "":
        return "Please enter a question.", ""
    result = answer_question(question)
    return result["answer"], "\n".join(result["sources"])

with gr.Blocks() as demo:
    gr.Markdown("# Company Knowledge Base Agent")
    gr.Markdown(
        "Ask questions about company policies and documents. "
        "Answers are generated from local indexed files only."
    )

    with gr.Row():
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g., What is the leave policy for new employees?",
            lines=2,
        )

    with gr.Row():
        answer = gr.Textbox(label="Answer (from retrieved context)", lines=12)
        sources = gr.Textbox(label="Source metadata", lines=12)

    ask_btn = gr.Button("Ask")

    ask_btn.click(
        fn=qa_interface,
        inputs=[question],
        outputs=[answer, sources],
    )

if __name__ == "__main__":
    demo.launch()
