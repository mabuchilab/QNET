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