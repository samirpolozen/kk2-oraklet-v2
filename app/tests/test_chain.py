from app.chain.pipeline import ai_pipeline
from app.chain.steps import LLMRunner, PromptBuilder, ResponseParser
from app.schemas import (
    LLMRunnerOutput,
    PromptBuilderInput,
)

    # Testar första steget i kedjan
    # Promtbuilder har i ansvar att bygga promten som skickas vidare till AI modellen
def test_prompt_builder_contains_question_and_stats():
    # Skapar testdata som motsvarar de PromtBuilder normalt får från API: et
    input_data = PromptBuilderInput(
        question="What is the mean value of temp_c?",
        stats={"temp_c": {"mean": 6.4, "max": 8.3}},
    )
    
    # Kör PromtBuilder steget och bygger en färdig promt
    result = PromptBuilder().invoke(input_data)
    
    # Verifierar att användarens fråga finns med i promten
    assert "What is the mean value of temp_c?" in result.prompt
    # Verifierar att statistikens kolumnnamn finns med
    assert "temp_c" in result.prompt
    # Verifierar att statistikvärdena har inkluderats i prompten.
    assert "mean" in result.prompt
    # Säkerställer att frågan följer med vidare i kedjan.
    assert result.question == input_data.question

    # Testar sista steget i kedjan
    # ResponseParser ska rensa bort prefix och städa modellns råoutput innan svaret skickas tillbaka till klienten via API: et
def test_response_parser_cleans_raw_model_output():
    raw_output = LLMRunnerOutput(
        question="What is the mean value of temp_c?",
        raw_output="Answer: The mean value of temp_c is 6.4.",
        model="test-model",
    )

    result = ResponseParser().invoke(raw_output)

    assert result.answer == "The mean value of temp_c is 6.4."
    assert result.question == raw_output.question
    assert result.model == "test-model"

# Testar hela Runnable-kedjan utan att använda den riktiga AI-modellen
# LLMRunner mockas så att testet blir kvickt, stabilt och oberoende av modellens svar
def test_pipeline_with_mocked_llm(monkeypatch):

    # Ersätter LLMRunner.invoke med ett fördefinierat svar, så testas kedjans logik utan att SmolLM laddas
    def fake_invoke(self, data):
        return LLMRunnerOutput(
            question=data.question,
            raw_output="Answer: Mocked answer.",
            model="mock-model",
        )

    monkeypatch.setattr(LLMRunner, "invoke", fake_invoke)

    chain_input = PromptBuilderInput(
        question="What is the mean value of temp_c?",
        stats={"temp_c": {"mean": 6.4}},
    )

    result = ai_pipeline.invoke(chain_input)

    # Verifierar att svaret har passerat genom hela kedjan
    assert result.answer == "Mocked answer."

    # Kollar om frågan följer med genom hela flödet
    assert result.question == chain_input.question

    # Till sist att modellinformationen följer med till slutresultatet
    assert result.model == "mock-model"