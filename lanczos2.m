function [y] = lanczos2(x,deltat,tc)
%
%Filtro Lanczos (modificado de Emery & Thomson, 1998 por
%                Joao Luiz Baptista de Carvalho em outubro de 2002)
%
% dados de entrada
%
% x                  = serie temporal a ser filtrada
% deltat             = intervalo de amostragem (unidade de tempo)
% tc                 = periodo de corte (unidade de tempo)
%
% dados de saida
%
% y                  = serie filtrada
%

wn=2*pi/(2*deltat); 
fc=1/tc;
wc=2*pi*fc;
wcwn=wc/wn;
%m=numero de componentes do filtro;
m=60;

%aumento do tamanho da serie para nao haver cortes na serie filtrada
n=length(x);
xx(1:n+2*m) = 0;
xx(m+1:n+m) = x;
xx(1:m) = x(m:-1:1);
xx(n+m+1:n+2*m) = x(n:-1:n-(m-1));


for k=1:m
   hk=sin(pi*k*wcwn)/(pi*k*wcwn);
   sigma=sin(pi*k/m)/(pi*k/m);
%  f(k)=1/2*sigma*hk;  (como est'a no livro do Emery)
   f(k)=sigma*hk;
end    
   
for n=m+1:length(x)+m

   somaf=0;
   for k=1:m
      somaf = somaf+f(k)*(xx(n-k)+xx(n+k));
   end    
%  y(n-m)=2*wcwn*(xx(n)+somaf);  (como est'a no livro do Emery)
   y(n-m)=wcwn*(xx(n)+somaf);

end 

y=y';
