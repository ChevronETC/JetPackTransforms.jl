using Documenter, DocumenterMarkdown, JetPackTransforms

makedocs(
    format = Markdown(),
    sitename = "JetPackTransforms"
)
cp("build/README.md", "../README.md", force=true)
