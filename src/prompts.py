# TODO: Add default options
from pydantic import BaseModel
from .pipeline import pipeline

class Prompt(BaseModel):
    template: str

    def __init__(self, template: str) -> None:
        super().__init__(template=template)

    def from_file(path: str) -> "Prompt":
        with open(path) as f:
            return Prompt(template=f.read())
    
    def fill(self, value: str, key: str = "text") -> "PromptPipeline":
        return Prompt(self.template.replace(f"{{{key}}}", value))
    
    def fill_pipeline(self, key: str = "text") -> "PromptPipeline":
        return PromptPipeline(template=self.template, key=key)

    def __str__(self) -> str:
        return self.template

@pipeline
def PromptPipeline(
    text: str, 
    template: str | None = None, 
    template_path: str | None = None,
    key: str = "text",
) -> str:
    if template is None:
        if template_path is None:
            raise ValueError("You must provide a prompt template or a prompt template path")
        with open(template_path) as f:
            template = f.read()
    else:
        if template_path is not None:
            raise ValueError("You must provide a prompt template or a prompt template path, not both")
    return Prompt(template.replace(f"{{{key}}}", text))

@pipeline
def PromptMulti(
    data: dict[str, str],
    template: str | None = None, 
    template_path: str | None = None,
) -> str:
    if template is None:
        if template_path is None:
            raise ValueError("You must provide a prompt template or a prompt template path")
        with open(template_path) as f:
            template = f.read()
    else:
        if template_path is not None:
            raise ValueError("You must provide a prompt template or a prompt template path, not both")
    return template.format(data)
