using Documenter
using ProximalPolicy

makedocs(
    sitename = "ProximalPolicy.jl",
    modules = [ProximalPolicy],
    pages = [
        "Home" => "index.md",
        "Environment Design" => "Environment_Design.md",
        "Implementation Details" => "implementation_details.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

deploydocs(
    repo = "github.com/jonathonbrown/ProximalPolicy.jl.git",
    push_preview = true,
)
