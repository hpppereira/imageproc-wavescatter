function [posX posY] = ipvd(vx,vy,xo,yo)
% Progressive Vector Diagram   ipdv.m
% 
% This script plots a 2D progressive vector diagram by using the Matlab 
% built-in quiver function. Two series of vector components vx, vy are 
% required. Initial position for plotting is (0,0) by default. 
% User can modify the initial position by entering xo, yo.
%
% Use:
% >> ipdv(vx,vy)
% or
% >> ipdv(vx,vy,Xo,Yo)
%
% So, in this example, vx and vy are arrays containing the coordinates of the vectors to plot
% they should have the same length.
%
% On the other hand, Xo, and Yo are coordinates of a single initial point (xo, yo) in the plane,
% not an array.
%
%
% Isaac M. M. INOGS, Trieste(Italy). November 24, 2009 @12h37:11
% Developed under Matlab(TM) version 7.1.0.246 (R14) Service Pack 3
%
if nargin < 2
   disp('two components vectors are required');
end
if nargin == 2
   disp('Default initial position used (0,0)');
   xo = 0;
   yo = 0;
end
if nargin == 3
   disp('X-position is updated. Y-position is 0 by default');
   yo = 0;
end
if nargin == 4
   disp('Initial position has been entered by the user')
end
n = length(vx);
innX = isnan(vx);
innY = isnan(vy);
vx(innX) = 0;
vy(innY) = 0;                      % NaNs replaced by Zeros so that Cumulative Sum can be computed (cumsum.m)
posX = cumsum([xo ; vx(:)]);
posY = cumsum([yo ; vy(:)]);
posX([isnan(1) ; innX(:)]) = NaN;  % NaNs re-inserted in their original locations.
posY([isnan(1) ; innY(:)]) = NaN;
quiver(posX(1:n),posY(1:n),vx(:),vy(:),0);
