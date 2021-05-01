function[aa]=alisa(zn1)

%alisa o espectro

%Este tipo de alisamento equivale a aplicar uma janela de hanning tantas vezes
%quanto desejado no dominio do tempo.  No dominio da frequencia isto equivale a
%convoluir a janela de hanning com o espectro.

%O numero de graus de liberdade eh dado por (2 * y), tal que

%                        log10(y)=0.4649*log10(x)+0.4251

%Y eh o somatorio dos quadrados dos pesos usados no alisamento e x eh o numero de
%vezes que a janela de hanning eh aplicada. (2 * y) eh chamado de Grau de Liberdade
%Equivalente (GLE)(Priestley, M.B. (1981) "Spectral Analysis and time series"
%Academic Press (ISBN 0-12-564922-3) pp 467.

%A relacao logaritimica acima foi ajustada numericamente ("least square fitting") 
%utilizando-se os pesos dos filtros quando se passa a hanning 1, 2, 3, e 4 vezes
%no espectro. Aplicando-se a janela de Hanning por 8 vezes, y = 6.9997 (~7) e o
%GLE ~ 14.  

reg=length(zn1);

%use esta rotina se x for maior do que 4

zzn1=zn1;

for i=1:8
   
   for j=2:reg-1;
      zzn1(j)=(zn1(j-1)+zn1(j)*2+zn1(j+1))/4;
   end

   zzn1(1)=(zn1(1)+zn1(1)*2+zn1(2))/4;
   zzn1(reg)=(zn1(reg)+zn1(reg)*2+zn1(reg-1))/4;
   zn1=zzn1;

end

%zzn1=zn1;

%aa=zzn1*sqrt(6.9997)';
aa=zzn1;

%use esta rotina se x for igual a 4 pois ela agiliza o alisamento

%for j=5:reg/2;
%  zzn1(j)=(zn1(j-4)+8*zn1(j-3)+28*zn1(j-2)+56*zn1(j-1)+70*zn1(j)+56*zn1(j+1)+28*zn1(j+2)+8*zn1(j+3)+zn1(j))/256;
%end

%  zzn1(1)=(zn1(4)+8*zn1(3)+28*zn1(2)+56*zn1(1)+70*zn1(1)+56*zn1(2)+28*zn1(3)+8*zn1(4)+zn1(5))/256;
% zzn1(2)=(zn1(3)+8*zn1(2)+28*zn1(1)+56*zn1(1)+70*zn1(2)+56*zn1(3)+28*zn1(4)+8*zn1(5)+zn1(6))/256;
%  zzn1(3)=(zn1(2)+8*zn1(1)+28*zn1(1)+56*zn1(2)+70*zn1(3)+56*zn1(4)+28*zn1(5)+8*zn1(6)+zn1(7))/256;
%  zzn1(4)=(zn1(1)+8*zn1(1)+28*zn1(2)+56*zn1(3)+70*zn1(4)+56*zn1(5)+28*zn1(6)+8*zn1(7)+zn1(8))/256;

%aa=zzn1*sqrt(6.9999)';

