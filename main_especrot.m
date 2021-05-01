% Processamento dos dados dos paths x e y
% com o espectro rotatorio

close all

xy = load('xy.txt');

dt = 1./30.0;

% x = diff(xy(500:1500,1));
% y = diff(xy(500:1500,2));

x = xy(500:1500,1);
y = xy(500:1500,2);

x_low_freq = lanczos2(x, dt, 2)
y_low_freq = lanczos2(y, dt, 2)

x = x - x_low_freq
y = y - y_low_freq

t = 1:length(x);

figure
plotyy(t, x, t, y)

% x = detrend(x1)
% y = detrend(y1)

% vetor progressivo
% [x3 y3] = ipvd(x1, y1);

% calculo do espectro cruzado
% [aa2] = espec2(x, y, dt);

% calculo do espectro rotatorio
[aa] = especrot(x, y, dt);


auto-espectro
figure
plot(aa(:,1),aa(:,2))
hold all
plot(aa(:,1),aa(:,3))
legend('x','y')
xlabel('Frequência (Hz)')
ylabel('Energia')
title('Auto-espectro')

% espectro horario e anti-horario
figure
plot(aa(:,1),aa(:,4))
hold all
plot(aa(:,1),aa(:,5))
legend('horario','anti-horario')
xlabel('Frequência (Hz)')
ylabel('Energia')
title('Espectro horario e anti-horario')

figure
plot(aa(:,1),aa(:,8))
legend('coef. rotacao')
xlabel('Frequência (Hz)')
ylabel('Energia')
title('Espectro horario e anti-horario')

% espectro de fase
% figure
% plotyy(aa2(:,1), aa2(:,7), aa2(:,1), aa2(:,8))

