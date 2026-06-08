# Reflektionsrapport 

## Säkerhet 
I detta projekt använder jag ingen extern API nyckel efetrsom språkmodellen körs lokalt via 'transformers.pipeline'. Hade jag använt en extern AI tjänst ex, OpenAI hade API nyckeln lagrats i en '.env' fil som hade lästs in som en miljövariabel. Ett fel som hade kunnat uppstå är att '.env' filen av misstag hade checkats in på GitHub skulle vem som helst kunna använda nyckeln och detta hade kunnat leda till obehörig användning.
Annan säkerhetsrisk är filuppladdning, API:t tar emot filer från användaren via endpointen /data/upload. Hade inte filerna kontrollerats skulle användaren kunna försöka ladda upp felaktiga eller oväntade filer. För att minska denna risk kontrollerar jag att filnamnet slutar på '.csv' innan ens Pandas försöker läsa innehållet. Jag hanterar även tomma file och fel teckenkodning genom att reunera tydliga 'HTTPException' felmeddelanden.
Exempel från min kod: 

```python
if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    content = await file.read()
    
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    try:
        decoded_content = content.decode("utf-8")
        df = pd.read_csv(StringIO(decoded_content))
     
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV file must use UTF-8 encoding.")
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file contains no readable data.")
    
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read CSV file.")
```

Promt injection är också en möjlig risk. Användarens fråga skickas in vidare till språkmodellen så skulle en användare kunna försöka manipulera modellen. I mitt projekt för tillfället finns inget skydd mot detta. Jag hade kunnat minska risken genom att bygga promten så att modellen instrueras att endast använda den statistik som skickas in från datasetet.
--- 

## Dataskydd (GDPR)

I mitt projekt nu faktiskt användaren ladda upp valfritt dataset. Om datasetet innehåller personuppgifter, namn, e-post, personnummer så innebär det att dem kan skickas vidare till språkmodellen som en del av promten. I detta projektet så lagras dock datasetet endast temporärt i minnet genom variabeln 'data.dataset'. Informationen sparas inte i en databas eller disk, det minskar iallafall risken för långvarig lagrig av personuppgifter.
Hade tjänsten däremot börjat användas i produktion hade fler åtgärder tagits till. Jag hade ex anonymisera personuppgifter eller filtrera bort de innan de skickas till modellen, sen hade autentisering, logghantering och tydliga rutiner för radering av data varit viktigt ur ett GDRP perspektiv.
---

## AI risker och ansvar

Under utvecklingen av detta projekt märkte jag tydligt att 'SmolLm' hade svårt att tolka statistik korrekt. Ett konkret exempel var när jag ställde frågan: "What is the highest temerature?" Trots att temperaturvärden fanns i datasetet svarade modellen ibland med värden från andra kolumner eller gav ett helt fel svar. Detta visar en viktig begränsning hos mindre språkmodeller. De kan generera svar som låter rimliga med som nästan alltid inte är rätt.
Denna 'SmolLM' modell och andra mindre modell har betydligt färre parametrar än större modeller som GPT4:a. Det betyder att den har svårare att resonera kring komplex information, följa instruktioner konsekvent och tolka statistik korrekt. Det blev en hel del hallucinationer där modellen drar fel slutsatser trots rätt information i promten.
För att öka tillförlitligheten i detta projekt har jag skrivit flera tester med pytest, Jag har testat både enskilda 'Runnable' steg och hela kedjan. Ett av testerna använder en mockad version av 'LLMRunner', där jag verkligen kan verifiera kedjans logik utan att vara beroende av språkmodellens svar. 
Exempel från mitt projekt: 

```python
def test_pipeline_with_mocked_llm(monkeypatch):

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

    assert result.answer == "Mocked answer."
    assert result.question == chain_input.question
    assert result.model == "mock-model"
  ```

Jag kan säkerställa att 'Runnable kedjan' fungerar genom att mocka modellen, även om språkmodellen skulle ge felaktigt svar eller ha dålig kvalite. Såhär kan jag testa applikationens logik separat från modelles kvalite. 
---

## Designval 

Jag valde att bygga detta projekt och lösningen med en tylig separation mellan API:t , datahanteringen och AI kedjan. Min tanke var att som i alla andra projekt att varje fil skulle ha ett tydligt ansvar. För 'endpoints' är filen 'main.py' ansvarig, 'data.py' ansvarar för lagring av datasetet i minnet, 'schemas.py' innehåller 'Pydantic' modellerna och 'chain/' innehåller själva Runnable kedjan.
Mitt viktigaste designval i projektet är 'Runnable mönstret'. Istället för att skriva all AI logik i en enda stor funktion delade jag upp flödet i tre steg:

- 'PromtBuilder' - bygger promten 
- 'LLMRunner' - anropar språkmodellen
- 'ResponseParser' -  putsar till modellens svar.
Genom att använda | - opeatorn kunde stegen kopplas ihop till en kedja på ett tydligt sätt såhär: 

```python
ai_pipeline = (
    PromptBuilder()
    | LLMRunner()
    | ResponseParser()
)
```
Jag byggde även en egen 'Runnable' grundklass: 

```python
def __or__(self, other: Any) -> "RunnableSequence":
    if isinstance(other, Runnable):
        return RunnableSequence.model_construct(first=self, second=other)

    if callable(other):
        wrapped = RunnableLambda.model_construct(
            func=other,
            name=getattr(other, "__name__", None),
        )
        return RunnableSequence.model_construct(first=self, second=wrapped)

    return NotImplemented
  ```
Kedjan kan ju skrivas mer läsbart: 

```python
PromptBuilder() | LLMRunner() | ResponseParser()
```
Istället för att manuellt behöva köra varje steg efter varandra.
Att använda 'Pydantic' var också ett medvetet val att ha me genom hela kedjan. Varje steg har tydlig input och output.
Ett exempel från mina modeller är:

```python
class PromptBuilderInput(BaseModel):
    question: str = Field(min_length=1)
    stats: dict[str, Any]


class PromptBuilderOutput(BaseModel):
    question: str
    prompt: str
```
Detta gjorde det lättare för mig att förstå vad varje steg tar emot och vad det skickar vidare. Risken för att fel data skickas mellan stegen minskas med. Det nog lätt största hindret var att få alla delar att fungera tillsammans, FastAPI, Pandas, Pydantic,Runnable kedjan och SmolLm som gav huvudvärk. Detta löste jag dock genom att bygga projektet stegvis. Först tog jag tag i att '/health' skulle fungera, sedan byggde jag '/data/upload', '/data/stats/' och till sist kopplade jag på AI kedjan med '/ai/ask'. En annan huvudvärk var att 'SmolLM' ibland gav felaktiga och oväntade svar. Jag la till 'ResponseParser' för att putsa till modellens råoutput innan svaret skickas tillbaka till användaren. 
Ex kod: 

```python
prefixes = ["Answer:", "A:", "Response:", "-", "*"]

for prefix in prefixes:
    if answer.lower().startswith(prefix.lower()):
        answer = answer[len(prefix):].strip()
```
Modellen blir inte perfekt ändå...men det gör API svaret renare och mer strukturerat.
För att sammanfatta allt valde jag denna design eftersom den gör koden mer uppdelad, letare att testa och enklare att bygga vidare på. Runnable kedjan gjorde också att jag kunde testa varje steg separat med pytest, vilket hade varit betydligt svårare om all logik låg i en enda stor funktion. 


