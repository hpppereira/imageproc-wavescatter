function [aa]=especrot(x,y,dt)

%calcula espectro rotatorio de uma serie de vetores
%autor:  Joao Luiz Baptista de Carvalho  (carvalho@univali.br)
%DADOS DE ENTRADA  
%                 x = vetor de componentes horizontais (potencia de 2)
%                 Y = vetor de componentes verticais   (potencia de 2)
%                 dt = intervalo de amostragem
%DADOS DE SAIDA
%MATRIZ CONTENDO: COLUNA 1 = frequencia
%                        2 = auto espectro da componente x
%                        3 = auto espectro da componente y
%                        4 = espectro horario
%                        5 = espectro anti-horario
%                        6 = espectro total
%                        7 = espectro da diferenca
%                        8 = coeficiente de rotacao
%                        9 = direcao da elipse
%                        10 = estabilidade da elipse
%                        11 = limite de confianca inferior do espectro total
%                        12 = limite de confianca superior do espectro total
%                        13 = intervalo de confianca para a estabilidade
%
%SUBROTINAS CHAMADAS
%               alisa
%               corthet2
%            


reg=length(x);
reg2=reg/2;
fc=1/(2*dt);
deltaf=fc*2/reg:fc*2/reg:fc;

%calculando a fft

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

%Alisa o espectro.

zn1 = alisa(zn1);
zn2 = alisa(zn2);
zn3 = alisa(zn3);
zn4 = alisa(zn4);
zn5 = alisa(zn5);
zn6 = alisa(zn6);
zn7 = alisa(zn7);
zn8 = alisa(zn8);

%monta os espectros rotatorios

const= dt/reg;

autn1 = const*(zn1+zn2);
autn2 = const*(zn3+zn4);
con1n2 = const*(zn5+zn6);
qdn1n2 = const*(zn7-zn8);
hora = 1/2*(autn1+autn2-2*qdn1n2);
antihora = 1/2*(autn1+autn2+2*qdn1n2);
total=(antihora+hora);
difs=-1/2*qdn1n2;
corot=difs./total;
phi=corthet2(autn1-autn2,2*con1n2)';
stab=1/4*(autn1.^2+autn2.^2-2*(autn1.*autn2-2*con1n2.^2))./(hora.*antihora);

%calcula intervalo de confianca para o espectro total (95%)

icinf = total*14/26.12;
icsup = total*14/5.63;

%calcula intervalo de confianca para a estabilidade (95%)

icest(1:reg2,1)=1-(0.05^(1/(14/2-1)));

%OBS:  Os intervalos de confianca dependem do Grau de Liberdade.  No caso usou-se GL=14.
%Se dejejar alterar o alisamento altere as linhas de comando.

%monta a matriz de saida

aa=[deltaf' autn1(1:reg2) autn2(1:reg2) hora(1:reg2) antihora(1:reg2) total(1:reg2) difs(1:reg2) corot(1:reg2) phi(1:reg2) stab(1:reg2) icinf(1:reg2) icsup(1:reg2) icest(1:reg2)];
