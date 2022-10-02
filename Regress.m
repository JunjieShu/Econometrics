function model = Regress(y,x,options)
%Regress
%   model = Regress(y,x) returns a structure array that saves different
%   types of estimators and statistics for OLS regression.
%   
%   Regress(y,x,options), specifies "Constant" ('Yes' or 'No'), "Table" 
%   ('Yes' or 'No') to show table or not, and level for confident level.
%
% Example
%   model = Regress(gp100m,weight,gear_ratio,Constant="Yes",Table="Yes",level=95);

%   References:
%       [1] Badi H. Baltagi (2011) Econometric 5th ed., Springer


arguments
    y (:,1) {mustBeNumeric,mustBeReal}
end

arguments (Repeating)
    x (:,1) {mustBeNumeric,mustBeReal}
end

arguments
    options.Constant (1,:) char {mustBeMember(options.Constant,{'Yes','No'})} = 'Yes'
    options.Table (1,:) char {mustBeMember(options.Table,{'Yes','No'})} = 'Yes'
    options.level {mustBeNumeric,mustBeReal} = 95
end

N = size(y,1);
K = nargin - 1;

model.depvar = inputname(1);

if nargin > 1
    model.indepvar = {};
    X = cell2mat(x);
    for n = 2 : nargin
        model.indepvar{end+1} = inputname(n);
    end
end

switch options.Constant
    case 'Yes'
        model.indepvar{end+1} = 'Cons';
        X(:,end+1) = 1;
        K = K + 1;
    case 'No'
end
%}


if nargin > 1
    X = cell2mat(x);
    switch options.Constant
        case 'Yes'
            model.indepvar = {'Cons'};
            for n = 2 : nargin
                model.indepvar{end+1} = inputname(n);
            end
            K = K + 1;
            X = [ones(N,1),X];
        case 'No'
            for n = 2 : nargin
                model.indepvar{end+1} = inputname(n);
            end
    end
end



% estimate independent variables

b = inv((X.'*X)) * X.'*y;

% get rss
y_estimate = X*b;
residuals = y - y_estimate;
rss = residuals.' * residuals;

% estimate variance
s_2 = rss/(N-K);
s = sqrt(s_2);


% regression statistics
V = s_2*(inv(X.'*X));
std_err = sqrt(diag(V));
t = b./std_err;
p_value_t = 1 - (tcdf(abs(t),N-K) - tcdf(-abs(t),N-K));

alpha = (1 - options.level/100);
lower_bound = b - abs(tinv(alpha/2,N-K)) * std_err;
upper_bound = b + abs(tinv(alpha/2,N-K)) * std_err;



% H0 of F-test is all beta's except intercept is 0, so the RRSS is
% equivalent to the RRS of regress y on ones.
RRSS = (y-mean(y)).'* (y-mean(y));
URSS = rss;

F = (RRSS - URSS)/(K-1) / s_2;
p_value_F = fcdf(F,K-1,N-K,'upper');

% centered / uncentered R^2
R2_centered = NaN;
R2_uncentered = NaN;
switch options.Constant
    case 'Yes'
        R2_centered = ( (y_estimate-mean(y)).'*(y_estimate-mean(y)) ) / ...
            ( (y-mean(y)).'*(y-mean(y)) );
        R2 = R2_centered;
        Adj_R2 = 1 - (N-1)/(N-K)*(1-R2);

        model.R2 = R2;
        model.Adj_R2 = Adj_R2;

    case 'No'
        R2_uncentered = ( y_estimate.'*y_estimate ) / ( y.'*y );
        R2 = R2_uncentered;
        Adj_R2 = 1 - (N-1)/(N-K-1)*(1-R2);

        model.R2 = R2;
        model.Adj_R2 = Adj_R2;
end



% Summarizing Table
Table = table;

TableColumnNames = {model.depvar 'Coefficient' 'Std. err.'...
    't' 'P>|t|' ['[',num2str(options.level),'% conf.'] 'interval]'};

Table.(TableColumnNames{1}) = model.indepvar';
Table.(TableColumnNames{2}) = b;
Table.(TableColumnNames{3}) = std_err;
Table.(TableColumnNames{4}) = t;
Table.(TableColumnNames{5}) = p_value_t;
Table.(TableColumnNames{6}) = lower_bound;
Table.(TableColumnNames{7}) = upper_bound;

% Table.Properties.RowNames=model.indepvar;

switch options.Table
    case 'Yes'
        disp(' ')
        disp(' ')
        disp(Table)
end


% results returned
model.N = N;
model.K = K;

model.b = b;
model.X = X;
model.y = y;
model.y_estimate = y_estimate;
model.residuals = residuals;
model.rss = rss;

model.s_2 = s_2;
model.s = s;
model.V = V;
model.mse = s_2; % mean square error; same thing, different name
model.rmse = s; % root of mean square error; same thing, different name



model.std_err = std_err;
model.t = t;
model.p_value_t = p_value_t;
model.F = F;
model.p_value_F = p_value_F;

model.R2_centered = R2_centered;
model.R2_uncentered = R2_uncentered;


model.Table = Table;


end