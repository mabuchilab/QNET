v 20110115 2
C 25500 63900 1 0 0 three_port_kerr_cavity-1.sym
{
T 26195 66995 5 8 0 0 0 0 1
device=ThreePortKerrCavity
T 26700 65700 5 8 1 1 0 0 1
refdes=C
T 26400 63500 5 10 1 0 0 0 1
Delta=Delta
T 26400 63300 5 10 1 0 0 0 1
chi=chi
T 26400 63100 5 10 1 0 0 0 1
kappa_1=kappa_1
T 26400 62900 5 10 1 0 0 0 1
kappa_2=kappa_2
T 26400 62700 5 10 1 0 0 0 1
kappa_3=kappa_3
T 26095 66795 5 8 0 0 0 0 1
params=Delta:real;chi:real;kappa_1:real;kappa_2:real;kappa_3:real
}
C 23500 66300 1 270 0 beamsplitter-1.sym
{
T 25600 64300 5 10 0 0 270 0 1
device=Beamsplitter
T 25200 64500 5 10 1 1 270 0 1
refdes=B1
}
C 27900 63500 1 0 0 beamsplitter-1.sym
{
T 29900 65600 5 10 0 0 0 0 1
device=Beamsplitter
T 29700 65200 5 10 1 1 0 0 1
refdes=B2
T 28900 65600 5 10 1 0 0 0 1
theta=theta
}
C 22700 64600 1 0 0 input-1.sym
{
T 22700 65500 5 10 0 0 0 0 1
device=IPAD
T 23000 65000 5 10 1 1 0 0 1
refdes=In2
T 22900 65200 5 10 1 0 0 0 1
pinseq=i2
}
C 30800 65000 1 180 0 output-1.sym
{
T 30800 64100 5 10 0 0 180 0 1
device=OPAD
T 30800 64600 5 10 1 1 180 0 1
refdes=Out1
T 30800 64300 5 10 1 0 180 0 1
pinseq=o1
}
C 26800 64300 1 0 0 phase-1.sym
{
T 27795 65795 5 8 0 0 0 0 1
device=Phase
T 28203 65185 5 8 1 1 0 0 1
refdes=Phase
T 28000 64200 5 10 1 0 0 0 1
phi=phi
}
N 27700 64800 27600 64800 4
N 25300 64800 26000 64800 4
N 23600 64800 24300 64800 4
N 24800 65700 24800 65300 4
N 26200 64400 26200 63800 4
N 26200 63800 29400 63800 4
N 29400 63800 29400 64300 4
T 22200 66900 8 10 1 0 0 0 1
params=Delta:real:50.0;chi:real:-0.26;kappa_1:real:20.0;kappa_2:real:20.0;kappa_3:real:10.0;theta:real:0.6435;phi:real:-1.39
T 26300 67200 8 10 1 0 0 0 1
module-name=And
C 24600 66600 1 270 0 input-1.sym
{
T 25500 66600 5 10 0 0 270 0 1
device=IPAD
T 25000 66300 5 10 1 1 270 0 1
refdes=In1
T 25200 66400 5 10 1 0 270 0 1
pinseq=i1
}
