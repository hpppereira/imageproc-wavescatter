function [aa]=espec2(x,y,dt)

%calcula espectro cruzado entre duas series reais
%DADOS DE ENTRADA  
%                 x = vetor de componentes horizontais (potencia de 2)
%                 Y = vetor de componentes verticais   (potencia de 2)
%                 dt = intervalo de amostragem
%DADOS DE SAIDA
%MATRIZ CONTENDO: COLUNA 1 = frequencia
%                        2 = auto espectro da componente x
%                        3 = auto espectro da componente y
%                        4 = co-espectro 
%                        5 = quad-espectro
%                        6 = amplitude do espectro cruzado
%                        7 = fase do espectro cruzado
%                        8 = espectro de coerencia
%                        9 = intervalo de conficanca inferior do espectro cruzado  
%                       10 = intervalo de conficanca superior do espectro cruzado  
%                       11 = intervalo de confianca da coerencia 
%SUBRROTINAS CHAMADAS

%                  alisa.m


reg=length(x);
reg2=reg/2;
%reg2=reg;
fc=1/(2*dt);
deltaf=fc*2/reg:fc*2/reg:fc;

%calcula a fft

x1=fft(x-mean(x));
y1=fft(y-mean(y));

%separa as componentes reais e imaginarias

z1=real(x1);
z2=imag(x1);
z3=real(y1);
z4=imag(y1);

zn1 = z1.*z1;
zn2 = z2.*z2;
zn3 = z3.*z3;
zn4 = z4.*z4;
zn5 = z1.*z3;
zn6 = z2.*z4;
zn7 = z2.*z3;
zn8 = z1.*z4;

%Alisa o espectro 

zn1 = alisa(zn1);
zn2 = alisa(zn2);
zn3 = alisa(zn3);
zn4 = alisa(zn4);
zn5 = alisa(zn5);
zn6 = alisa(zn6);
zn7 = alisa(zn7);
zn8 = alisa(zn8);

%monta os espectros cruzados

const= 2*dt/reg;
autn1 = const*(zn1+zn2);
autn2 = const*(zn3+zn4);
con1n2 = const*(zn5+zn6);
qdn1n2 = const*(zn7-zn8);
fasen1n2 = atan(qdn1n2./con1n2)*180/pi;
ampln1n2 = sqrt(qdn1n2.^2+con1n2.^2);
coerenc = (con1n2.^2+qdn1n2.^2)./(autn1.*autn2);

%calcula intervalo de confianca para o espectro cruzado (95%)

icinf = con1n2*14/26.12;
icsup = con1n2*14/5.63;

%calcula intervalo de confianca para a coerencia (95%)

iccoer(1:reg2,1)=1-(0.05^(1/(14/2-1)));

%OBS:  Os intervalos de confianca dependem do Grau de Liberdade.  No caso usou-se GL=14.
%Se dejejar alterar o alisamento altere as linhas de comando.

aa = [deltaf' autn1(1:reg2) autn2(1:reg2) con1n2(1:reg2) qdn1n2(1:reg2) ampln1n2(1:reg2) ...
      fasen1n2(1:reg2) coerenc(1:reg2) icinf(1:reg2) icsup(1:reg2) iccoer(1:reg2)];

