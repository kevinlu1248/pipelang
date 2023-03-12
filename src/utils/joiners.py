from ..pipeline import pipeline

@pipeline
def SimpleJoinerPipeline(
    text: str,
    joiner: str = "\n\n",
) -> str:
    return joiner.join(text)
