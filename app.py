import gradio as gr
import os
from huggingface_hub import InferenceClient

# جلب المفتاح من إعدادات السيرفر بأمان
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-72B-Instruct", token=hf_token)

def ai_teacher(grade, subject, question):
    if not question.strip():
        return "من فضلك اكتب سؤالك أولاً يا بطل! 😊"

    # بناء البرومبت التعليمي الذكي بناءً على اختيارات الطالب
    system_message = f"أنت معلم خبير ومتخصص في مادة {subject} لطلاب الصف {grade}. أجب دائماً باللغة العربية، بأسلوب سهل، مشوق، ومناسب لعمر الطالب. استخدم الرموز التعبيرية لجعل الإجابة ممتعة."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]

    try:
        response = ""
        # استدعاء الموديل مع تفعيل خاصية البث (Streaming) وتحقق الحماية
        for message in client.chat_completion(
            messages,
            max_tokens=1000, 
            stream=True,
        ):
            if hasattr(message, 'choices') and len(message.choices) > 0:
                token = message.choices[0].delta.content
                if token is not None:
                    response += token
        
        return response if response else "المعلم يفكر.. حاول إعادة صياغة السؤال."
        
    except Exception as e:
        return f"عذراً، المعلم مشغول حالياً! (الخطأ: {str(e)})"

# --- تصميم واجهة المستخدم ---
# قمنا بإصلاح السمة هنا لحل خطأ الـ TypeError
with gr.Blocks(theme=gr.themes.Soft(), css="* {direction: rtl;}") as demo:
    gr.Markdown("""
    # 🏫 منصة المعلم الذكي للأجيال
    ### أهلاً بك في مدرستك الذكية! اختر صفك ومادتك واسأل ما تشاء.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            grade_dropdown = gr.Dropdown(
                choices=["الأول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع"],
                label="اختر الصف الدراسي",
                value="الخامس"
            )
            subject_dropdown = gr.Dropdown(
                choices=["الرياضيات", "العلوم", "اللغة الإنجليزية", "الجغرافيا", "اللغة العربية"],
                label="اختر المادة",
                value="الرياضيات"
            )
        
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="اكتب سؤالك هنا",
                placeholder="مثال: كيف يتكون المطر؟ أو ما هو ناتج 5 ضرب 6؟",
                lines=4
            )
            submit_btn = gr.Button("اسأل المعلم 🚀", variant="primary")

    with gr.Row():
        answer_output = gr.Markdown(label="شرح المعلم")

    # ربط الأحداث
    submit_btn.click(
        fn=ai_teacher,
        inputs=[grade_dropdown, subject_dropdown, question_input],
        outputs=answer_output
    )

    gr.Markdown("---")
    gr.Markdown("Designed with ❤️ for Education by Teacher Mohammed")

if __name__ == "__main__":
    demo.launch()