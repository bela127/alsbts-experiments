clear all
KpOutput=[1.0];
Ts=1e-3; fc=1/Ts;
stop_time = 600;
log_time = 0.2;
Vdc=650;
Vref=325;fref_st=50;
Iref=2;
Pnom=50e3;
%----control desk variables
EN_ST=0;       %enable ST PWM
RSTanalog=0;

%----Control PulseGenerator with this trigger

%----

%======ST control Parameters=====================
   %--outer voltage control loop-----------------
kpv=0.2; %0.25; Jason Exercise % 0.5; Jason transients % proportional gain of voltage control loop
kiv=250;   % integral gain of voltage cntrol loop
    %------inner current control loop------------------
kpi_ST=1.25; % 2.5; %Jason transients % proportional gain of current control loop
kii_ST=250;


%% ST parameters
Rc=0.1; %Filter resistance
Lc=1e-3; %Filter inductance
Cf=10e-6; %Filter capacitance
Rf=2; % Filter Damping

Rg=1; %Grid resistance
Lg=Rc/(2*2*pi*fref_st); % Grid inductance

%% Load parameter
Kpload=1;
Kqload=2;

%------PLL parameters----------------------
PLL_D=1/sqrt(2);
PLL_Tsettle=0.7;
Vamp=230*sqrt(2);
PLL_kpi=2*4.6 /PLL_Tsettle;
PLL_ki=2*PLL_Tsettle*(PLL_D^2) / 4.6;

%------digital PR controller----------------
delta=0.05;
resonant=tf([2*2*pi*50*delta 0], [1 2*2*pi*50*delta (2*pi*50)^2]);
resonant_d=c2d(resonant, Ts);

resonant_i=tf([1 0], [1 0 (2*pi*50)^2]);
resonant_id=c2d(resonant_i, Ts);


%%% LoadShedding %%%
LoadCurtailment=0.9;
LoadShedding=0;
VoltageStep=0;
DroopCurr=1/Vref;
MaxCurr=200;
LimCur=0;

%%% Sensitivities %%%
VARV=0;
VARF=0;

%------digital filters---------------------
LEF.wf=2*pi*5;
LPF.wf=2*pi*450;

LEF.a=(tan(pi/4-pi/16))^2;
lef=tf([1/LEF.wf 1], [LEF.a/LEF.wf 1]);
lefz=c2d(lef,Ts,'Tustin');

LPF.D=1/sqrt(2);
lpf=tf([LPF.wf^2], [1 2*LPF.D*LPF.wf LPF.wf^2]);
lpfz=c2d(lpf,Ts,'Tustin');

%%% ST starting filter
STstart=tf(1, [1.5 1]);
STstartz=c2d(STstart,Ts,'Tustin');

%%% filter
Filter=tf(1, [0.02 1]);
Filter_d=c2d(Filter,Ts,'Tustin');

