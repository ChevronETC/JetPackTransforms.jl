using Documenter, JetPackTransforms

makedocs(sitename = "JetPackTransforms", modules=[JetPackTransforms])

deploydocs(
    repo = "github.com/ChevronETC/JetPackTransforms.jl.git",
)

