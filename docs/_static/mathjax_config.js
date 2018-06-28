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
        trace: ['{\\operatorname{tr}}', 0],
        tr: ['{\\operatorname{tr}}', 0],
        Tr: ['{\\operatorname{tr}}', 0],
        diag: ['{\\operatorname{diag}}', 0],
        fwhm: ['{\\operatorname{fwhm}}', 0],
        abs: ['{\\operatorname{abs}}', 0],
        pop: ['{\\operatorname{pop}}', 0],
        rot: ['{\\operatorname{rot}}', 0],
        SLH: ['{\\operatorname{SLH}}', 0],
        aux: ['{\\text{aux}}', 0],
        opt: ['{\\text{opt}}', 0],
        tgt: ['{\\text{tgt}}', 0],
        init: ['{\\text{init}}', 0],
        avg: ['{\\left\\langle#1\\right\\rangle}', 1],
        bra: ['{\\left\\langle#1\\right\\vert}', 1],
        ket: ['{\\left\\vert#1\\right\\rangle}', 1],
        Bra: ['{\\left\\langle#1\\right\\vert}', 1],
        bra: ['{\\langle#1\\vert}', 1],
        Braket: ['{\\left\\langle #1\\vphantom{#2} \\mid #2\\vphantom{#1}\\right\\rangle}', 2],
        Ket: ['{\\left\\vert#1\\right\\rangle}', 1],
        ket: ['{\\vert#1\\rangle}', 1],
        mat: ['{\\mathbf{#1}}', 1],
        op: ['{\\hat{#1}}', 1],
        Op: ['{\\hat{#1}}', 1],
        dd: ['{\\\,\\text{d}}', 0],
        daggered: ['{^{\\dagger}}', 0],
        transposed: ['{^{\\text{T}}}', 0],
        Liouville: ['{\\mathcal{L}}', 0],
        DynMap: ['{\\mathcal{E}}', 0],
        identity: ['{\\mathbf{1}}', 0],
        Norm: ['{\\lVert#1\\rVert}', 1],
        Abs: ['{\\left\\vert#1\\right\\vert}', 1],
        Avg: ['{\\left<#1\\right>}', 1],
        AbsSq: ['{\\left\\vert#1\\right\\vert^2}', 1],
        Re: ['{\\mathfrak{Re}}', 0],
        Im: ['{\\mathfrak{Im}}', 0],
        }
    },
});
