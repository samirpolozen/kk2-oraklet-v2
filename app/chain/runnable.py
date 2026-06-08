from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, SerializeAsAny

# Typvariabler för input, output och data mellan stegen
I = TypeVar("I")
M = TypeVar("M")
O = TypeVar("O")


class Runnable(BaseModel, Generic[I, O]):
  # Basklass för ett steg i kedjan
  # Varje steg tar emot I och returnerar O
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None

    def invoke(self, data: I) -> O:
        # Måste tillämpas av subklasser
        raise NotImplementedError("Subclasses must implement invoke().")

    
    def __or__(self, other: Any) -> "RunnableSequence":
        # Gör så att vi kan skriva step1 | step2
        if isinstance(other, Runnable):
            return RunnableSequence.model_construct(first=self, second=other)

        if callable(other):
            # Wrappa funktionen så den fungerar som ett steg
            wrapped = RunnableLambda.model_construct(
                func=other,
                name=getattr(other, "__name__", None),
            )
            return RunnableSequence.model_construct(first=self, second=wrapped)

        return NotImplemented


class RunnableLambda(Runnable[I, O]):
    # Gör om en vanlig funktion till ett Runnable steg
    func: Callable[[I], O]

    def invoke(self, data: I) -> O:
        return self.func(data)


class RunnableSequence(Runnable[I, O], Generic[I, M, O]):
    # Här körs två steg efter varandra
    first: SerializeAsAny[Runnable[I, M]]
    second: SerializeAsAny[Runnable[M, O]]

    def invoke(self, data: I) -> O:
        intermediate = self.first.invoke(data)
        return self.second.invoke(intermediate)