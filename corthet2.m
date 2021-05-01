function [theta]= theta(a,b)
 
%     REDUZ AS DIRECOES DOS VETORES PARA O NORTE VERDADEIRO.
%     SOLUCAO ADAPTADA PARA O CALCULO DO ESPECTRO ROTATORIO.

%AUTOR:  JOAO LUIZ BAPTISTA DE CARVALHO (carvalho@cttmar.univali.rct-sc.br)

%     PARAMETROS DE ENTRADA
%     A,B ............... ORDENADAS DAS COMPONENTES

for i=1:length(a);      
   
    if (a(i)~=0)

       theta(i)=1/2*atan(abs(b(i))/abs(a(i)))*180/pi;      
    
       if (b(i)>0)
          if (a(i)>0)
             theta(i)=90-theta(i);
          else
             theta(i)=theta(i);   
          end
       elseif (b(i)<0)
          if (a(i)>0)
             theta(i)=90+theta(i);
          else
             theta(i)=180-theta(i);   
          end
       elseif (b(i)==0)
          if (a(i)<0)   
             theta(i)=180;
          else
             theta(i)=90;
          end
       end
          
else
       
       if (b(i)>0)
          theta(i)=45;
       elseif (b(i)<0)
          theta(i)=135;
       elseif (b(i)==0)
             theta(i)=0;
       end
             
    end      
          
end