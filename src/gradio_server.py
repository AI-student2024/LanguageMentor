import os
import gradio as gr
from agents.conversation_agent import ConversationAgent
from agents.scenario_agent import ScenarioAgent
from agents.model import ModelManager
from utils.logger import LOG

# 定义模型列表
ollama_models = ["llama3.1:8b-instruct-q4_0", "llama3.1:8b-instruct-q8_0"]
openai_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]

# 创建对话代理实例，无需在此处指定模型，稍后根据用户选择动态传递
conversation_agent = ConversationAgent()

# 定义场景代理的选择与调用，无需在此处指定模型，稍后根据用户选择动态传递
agents = {
    "job_interview": ScenarioAgent("job_interview"),
    "hotel_checkin": ScenarioAgent("hotel_checkin"),
    "salary_negotiation": ScenarioAgent("salary_negotiation"),
    "renting": ScenarioAgent("renting"),
    "airportBaggage_checkin": ScenarioAgent("airportBaggage_checkin")
}

# 处理用户对话的函数，动态传递模型名称和类型
def handle_conversation(user_input, chat_history, model_name, model_type):
    # 使用 ModelManager 根据用户选择的模型名称和类型动态创建模型
    model_manager = ModelManager(model_name=model_name)
    model_manager.model_type = model_type
    selected_model = model_manager.create_model()

    # 更新 conversation_agent 的模型
    conversation_agent.set_model(selected_model)
    bot_message = conversation_agent.chat_with_history(user_input)
    LOG.info(f"[ChatBot]: {bot_message}")
    return bot_message

# 处理场景代理的函数，动态传递模型名称和类型
def handle_scenario(user_input, chat_history, scenario, model_name, model_type):
    # 使用 ModelManager 根据用户选择的模型名称和类型动态创建模型
    model_manager = ModelManager(model_name=model_name)
    model_manager.model_type = model_type
    selected_model = model_manager.create_model()

    # 更新场景代理的模型
    agents[scenario].set_model(selected_model)
    bot_message = agents[scenario].chat_with_history(user_input)
    LOG.info(f"[ChatBot]: {bot_message}")
    return bot_message

# 获取场景介绍的函数
def get_scenario_intro(scenario):
    try:
        with open(f"content/page/{scenario}.md", "r", encoding="utf-8") as file:
            scenario_intro = file.read().strip()
    except FileNotFoundError:
        scenario_intro = "该场景的介绍尚未提供。"
    return scenario_intro

# 根据选择的模型类型更新模型名称列表
def update_model_list(model_type):
    if model_type == "ollama":
        return ollama_models
    elif model_type == "openai":
        return openai_models
    else:
        return []

# 根据选择的模型类型更新模型名称列表
def change_model_list(model_type):
    # 返回新的模型列表来覆盖 Dropdown 的选项
    return {
        "choices": update_model_list(model_type),  # 更新模型列表
        "value": None  # 清空当前的选中值，避免错误选项
    }

# Gradio 界面构建
with gr.Blocks(title="LanguageMentor 英语私教") as language_mentor_app:
    with gr.Tab("场景训练"):
        gr.Markdown("## 选择一个场景完成目标和挑战")

        # 模型类型选择
        model_type_radio = gr.Radio(
            choices=["ollama", "openai"],
            label="选择模型类型",
            value="ollama"  # 默认值
        )

        # 模型名称选择 (动态更新)
        model_name_dropdown = gr.Dropdown(
            choices=ollama_models,  # 默认使用 Ollama 的模型列表
            label="选择模型名称",
            value=None,  # 初始时没有选中的模型
            allow_custom_value=False  # 禁止用户输入自定义模型名称
        )

        # 根据模型类型的选择，动态更新模型名称列表
        model_type_radio.change(
            fn=change_model_list,
            inputs=[model_type_radio],
            outputs=[model_name_dropdown]
        )

        scenario_radio = gr.Radio(
            choices=[
                ("求职面试", "job_interview"),
                ("酒店入住", "hotel_checkin"),
                ("薪资谈判", "salary_negotiation"),
                ("租房", "renting"),
                ("机场托运", "airportBaggage_checkin")
            ],
            label="场景"
        )

        scenario_intro = gr.Markdown()
        scenario_chatbot = gr.Chatbot(placeholder="选择场景后开始对话吧！", height=600)

        def start_new_scenario_chatbot(scenario, model_name, model_type):
            initial_ai_message = agents[scenario].start_new_session()
            return gr.Chatbot(value=[(None, initial_ai_message)], height=600)

        scenario_radio.change(
            fn=lambda s, model_name, model_type: (get_scenario_intro(s), start_new_scenario_chatbot(s, model_name, model_type)),
            inputs=[scenario_radio, model_name_dropdown, model_type_radio],
            outputs=[scenario_intro, scenario_chatbot]
        )

        gr.ChatInterface(
            fn=handle_scenario,
            chatbot=scenario_chatbot,
            additional_inputs=[scenario_radio, model_name_dropdown, model_type_radio],
            retry_btn=None,
            undo_btn=None,
            clear_btn="清除历史记录",
            submit_btn="发送"
        )

    with gr.Tab("对话练习"):
        gr.Markdown("## 练习英语对话")

        conversation_chatbot = gr.Chatbot(placeholder="想和我聊什么话题都可以，记得用英语哦！", height=800)

        gr.ChatInterface(
            fn=handle_conversation,
            chatbot=conversation_chatbot,
            additional_inputs=[model_name_dropdown, model_type_radio],
            retry_btn=None,
            undo_btn=None,
            clear_btn="清除历史记录",
            submit_btn="发送"
        )

if __name__ == "__main__":
    # 启动 Gradio 应用并共享
    language_mentor_app.launch(share=True, server_name="0.0.0.0")
