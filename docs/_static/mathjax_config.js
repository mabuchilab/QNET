MathJax.Hub.Register.StartupHook("TeX Jax Ready",function () {
    var TEX = MathJax.InputJax.TeX;
    var MML = MathJax.ElementJax.mml;
    TEX.Definitions.macros.cancel =
        ["myCancel",MML.NOTATION.UPDIAGONALSTRIKE];
    TEX.Definitions.macros.bcancel =
        ["myCancel",MML.NOTATION.DOWNDIAGONALSTRIKE];
    TEX.Parse.Augment({
        myCancel: function (name,notation) {
            var mml = this.ParseArg(name);

            this.Push(MML.menclose(mml).With({notation:notation}));
        }
    });
});
MathJax.Hub.Config({
    jax: ["input/TeX","output/SVG"],
    TeX: {
        extensions: [
            "AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"
        ],
        Macros: {
        bra: ['{\\langle#1\\vert}', 1],
        ket: ['{\\vert#1\\rangle}', 1],
        }
    },
});
