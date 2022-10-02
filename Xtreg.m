function model = Xtreg(panelvar,timevar,y,x,options)
%Xtreg
%   model = Xtreg(panelvar,timevar,y,x) retruns a structure array that
%   saves different types of estimators and statistics for the Model
%   specified.
%
%   model = Xtreg(panelvar,timevar,y,x,options), specifies "Model" ('be',
%   'fe' or 're') and "Constant" ('Yes' or 'No'; only allowed for MLE 
%   method), "Table" ('Yes' or 'No') to show table or not, and level for 
%   confident level.
%
% Example
%   be = Xtreg(FN, YR, I, F, C, Model='be', Constant='Yes',Table='Yes',level=95);
%   fe = Xtreg(FN, YR, I, F, C, Model='fe', Constant='Yes',Table='Yes',level=95);
%   re = Xtreg(FN, YR, I, F, C, Model='re', Constant='Yes',Table='Yes',level=95);

%   References:
%       [1] Badi H. Baltagi (2021) Econometric analysis of panel data.
%           6th ed., Springer


arguments
    panelvar (:,1) {mustBeNumeric,mustBeReal}
    timevar (:,1) {mustBeNumeric,mustBeReal}
    y (:,1) {mustBeNumeric,mustBeReal}
end

arguments (Repeating)
    x (:,1) {mustBeNumeric,mustBeReal}
end

arguments
    options.Model (1,:) char {mustBeMember(options.Model,{'re','be','fe'})} = 're'
    options.Constant (1,:) char {mustBeMember(options.Constant,{'Yes','No'})} = 'Yes'
    options.Table (1,:) char {mustBeMember(options.Table,{'Yes','No'})} = 'Yes'
    options.level {mustBeNumeric,mustBeReal} = 95
end

model.depvar = inputname(3);

if nargin > 1
    switch options.Constant
        case 'Yes'
            model.indepvar = {'Cons'};
        case 'No'
            model.indepvar = {};
    end

    X = cell2mat(x);
    for n = 4 : nargin
        model.indepvar{end+1} = inputname(n);
    end
end



i = unique(panelvar);
t = unique(timevar);

T = size(t,1);
N = size(i,1);
K = nargin - 3; % in Xtreg, K is the number of X's, excluding constant


iota_NT = ones(N*T,1);
iota_T = ones(T,1);
I_N = eye(N);
I_NT = eye(N*T);

switch options.Constant
    case 'Yes'
        Z = [iota_NT, X];
    case 'No'
        Z = X;
end
Z_mu = kron(I_N, iota_T);


P = Z_mu * inv(Z_mu.'*Z_mu) * Z_mu.';
Q = I_NT - P;

Py = P*y;
Qy = Q*y;
PX = P*X;
QX = Q*X;
PZ = P*Z;
QZ = Q*Z;


switch options.Model
    case 'be'
        %%% Between
        [~, ia] = unique(Py,'stable');
        yi_dot = Py(ia);
        Zi_dot = PZ(ia,:);

        delta = regress(yi_dot,Zi_dot);
        b = delta(2:end);

        %%% get rss
        y_estimate = Zi_dot*delta;
        residuals = yi_dot - y_estimate;
        rss = residuals'*residuals;

        %%% estimate variance
        s_2 = rss/(N-K-1);
        s = sqrt(s_2);
        sigma_1_2 = s_2;  % sigma_1_2 = (sigma_mu_2 + sigma_v_2/T)
        sigma_1 = s;

        %%% regression statistics'
        V = s_2*(inv(Zi_dot.'*Zi_dot));
        std_err = diag((V).^(1/2));

        t = delta./std_err;
        p_value_t = 1 - (tcdf(abs(t),N-K) - tcdf(-abs(t),N-K));

        alpha = (1 - options.level/100);
        lower_bound = delta - abs(tinv(alpha/2,N-K-1)) * std_err;
        upper_bound = delta + abs(tinv(alpha/2,N-K-1)) * std_err;

        % H0 of F-test is all beta's except intercept is 0, so the RRSS is
        % equivalent to the RRS of regress yi_dot on ones.
        RRSS = (yi_dot-mean(yi_dot)).'* (yi_dot-mean(yi_dot));
        URSS = rss;

        F = (RRSS - URSS)/(K) / s_2;
        p_value_F = fcdf(F,K,N-K-1,'upper');


        % R^2  % see Stata pdf [xt] "Methods and formulas"
        temp = corr([(X-PX)*b, y-Py]);
        R2_within = temp(1,2)^2;

        R2_between = ( (y_estimate-mean(yi_dot))'*(y_estimate-mean(yi_dot)) ) /...
            ( (yi_dot-mean(yi_dot))'*(yi_dot-mean(yi_dot)) );

        temp = corr([X*b, y]);
        R2_overall = temp(1,2)^2;


        % Summarizing Table
        Table = table;

        TableColumnNames = {model.depvar 'Coefficient' 'Std. err.'...
            't' 'P>|t|' ['[',num2str(options.level),'% conf.'] 'interval]'};

        Table.(TableColumnNames{1}) = model.indepvar';
        Table.(TableColumnNames{2}) = delta;
        Table.(TableColumnNames{3}) = std_err;
        Table.(TableColumnNames{4}) = t;
        Table.(TableColumnNames{5}) = p_value_t;
        Table.(TableColumnNames{6}) = lower_bound;
        Table.(TableColumnNames{7}) = upper_bound;

        %         temp = Table(2:end,:);
        %         Table(end,:) = Table(1,:);
        %         Table(1:end-1, :) = temp;

        switch options.Table
            case 'Yes'
                disp(' ')
                disp(' ')
                disp(Table)
        end


        % results returned
        model.delta = delta;
        model.b = b;
        model.y_estimate = y_estimate;
        model.residuals = residuals;
        model.rss = rss;

        model.V = V;
        model.s_2 = s_2;
        model.s = s;
        model.sigma_1_2 = sigma_1_2;
        model.sigma_1 = sigma_1;
        model.std_err = std_err;

        model.t = t;
        model.p_value_t = p_value_t;
        model.F = F;
        model.p_value_F = p_value_F;
        model.R2_within = R2_within;
        model.R2_between = R2_between;
        model.R2_overall = R2_overall;

        model.Table = Table;

    case 'fe'
        %%% fix-effect / Within / LSDV
        b = regress(Qy,QX);
        a = mean(y) - mean(X)*b;
        mu = Z_mu \ (Py - a - PX*b); %P(.) is a operation representing averaging over time
        delta = [a;b];

        %%% get rss
        y_estimate = Z*delta + Z_mu*mu;
        residuals = y - y_estimate;
        rss = residuals'*residuals;

        %%% estimate variance
        s_2 = rss/(N*T-N-K);
        s = sqrt(s_2);  % this is also sigma_v
        sigma_v_2 = s_2;
        sigma_v = s;
        sigma_mu = sqrt(var(mu));
        sigma_mu_2 = sigma_mu^2;
        rho = sigma_mu_2/(sigma_mu_2+sigma_v_2);

        %%% regression statistics
        V_b = sigma_v_2 * inv((X'*QX));   % P 17
        V_a = mean(X) * V_b * mean(X)' + sigma_v_2/(N*T); % P 17, derived from formula (2.11)
        Cov_a_b = mean(X)*V_b;
        V = [V_a, Cov_a_b; Cov_a_b' V_b];

        std_err = [sqrt(V_a);   (diag(sigma_v_2 * inv((X'*QX)))).^(1/2)];

        t = delta./std_err;
        p_value_t = 1 - (tcdf(abs(t),N-K) - tcdf(-abs(t),N-K));

        alpha = (1 - options.level/100);
        lower_bound = delta - abs(tinv(alpha/2,N*T-N-K)) * std_err;
        upper_bound = delta + abs(tinv(alpha/2,N*T-N-K)) * std_err;

        % H0 of F-test is all b's are 0, so the RRSS is
        % equivalent to the RRS of regress y on dummies (totally N, including intercept).
        [~,~,e,~] = regress(y,Z_mu);
        RRSS = e.'* e;
        URSS = rss;

        F = (RRSS - URSS)/K / s_2;
        p_value_F = fcdf(F,K,N*T-N-K,'upper');

        % H0 of F-test is all mu's are 0 (N-1 mu's), so the RRSS is
        % equivalent to the RRS of OLS without mu.
        [~,~,e,~] = regress(y,Z);
        RRSS = e'*e;
        URSS = rss;

        F_mu = ( (RRSS - URSS)/(N-1) ) / ((URSS)/(N*T-N-K));
        p_value_F_mu = fcdf(F_mu,N-1,N*T-N-K,'upper');

        % R^2  % see Stata pdf [xt] "Methods and formulas"
        R2_within = ((QX*b-mean(Qy))'*(QX*b-mean(Qy))) / ((Qy-mean(Qy))'*(Qy-mean(Qy)));

        temp = corr([PX*b, Py]);
        R2_between = temp(1,2)^2;

        temp = corr([X*b, y]);
        R2_overall = temp(1,2)^2;

        % Summarizing Table
        Table = table;

        TableColumnNames = {model.depvar 'Coefficient' 'Std. err.'...
            't' 'P>|t|' ['[',num2str(options.level),'% conf.'] 'interval]'};

        Table.(TableColumnNames{1}) = model.indepvar';
        Table.(TableColumnNames{2}) = delta;
        Table.(TableColumnNames{3}) = std_err;
        Table.(TableColumnNames{4}) = t;
        Table.(TableColumnNames{5}) = p_value_t;
        Table.(TableColumnNames{6}) = lower_bound;
        Table.(TableColumnNames{7}) = upper_bound;

        switch options.Table
            case 'Yes'
                disp(' ')
                disp(' ')
                disp(Table)
        end


        % results returned
        model.b = b;
        model.a = a;
        model.mu = mu;
        model.delta = delta;

        model.y_estimate = y_estimate;
        model.residuals = residuals;
        model.rss = rss;
        model.s_2 = s_2;
        model.s = s;
        model.sigma_v_2 = sigma_v_2;
        model.sigma_v = sigma_v;
        model.sigma_mu = sigma_mu;
        model.sigma_mu_2 = sigma_mu_2;
        model.rho = rho;

        model.V = V;
        model.std_err = std_err;
        model.t = t;
        model.p_value_t = p_value_t;
        model.F = F;
        model.p_value_F = p_value_F;
        model.F_mu = F_mu;
        model.p_value_F_mu = p_value_F_mu;
        model.R2_within = R2_within;
        model.R2_between = R2_between;
        model.R2_overall = R2_overall;
        model.Table = Table;


    case 're'
        %%% random effect
        % We use GLS to estimate beta; but to do this, we must first
        % estimate \Omega, the variance-covariance matrix.

        % calculating sigma_1 and sigma_v (sigma_mu_2 = (sigma_1_2 - sigma_v_2)/T
        %{
            In Badi's textbook, we just need to use regression residuals to replace u
            in Best Quadratic Unbiased estimator, without corrected by degree of
            freedom, but these does not same result as in Eviews. Check this in the
            futrue.

        % WalHus (Wallace and Hussain)
        [~,~,e,~] = regress(y,Z);

        sigma_1_2 = e'*P*e/trace(P);
        sigma_v_2 = e'*Q*e/trace(Q);
        sigma_mu_2 = (sigma_1_2 - sigma_v_2)/T;

        sigma_v = sqrt(sigma_v_2);
        sigma_mu = sqrt(sigma_mu_2);

        % Amemiya: LSDV residuals

        b = inv(X.'*QX)*X.'*Qy;
        a = mean(y) - mean(X)*b; 
        e = y - a*iota_NT - X*b;
        
        sigma_1_2 = e'*P*e/trace(P);
        sigma_v_2 = e'*Q*e/trace(Q);
        sigma_mu_2 = (sigma_1_2 - sigma_v_2)/T;
        
        sigma_v = sqrt(sigma_v_2);
        sigma_mu = sqrt(sigma_mu_2);
        %}

        % Swamy-Arora
        [~,~,e,~] = regress(Qy,QX); % within regression
        sigma_v_2 = e'*e/(N*(T-1)-K);
        sigma_v = sqrt(sigma_v_2);

        [~,~,e,~] = regress(Py,PZ); % between regression
        sigma_1_2 = e'*e/(N-K-1);
        sigma_1 = sqrt(e'*e/(N-K-1));

        sigma_mu_2 = (sigma_1_2 - sigma_v_2)/T;
        sigma_mu = sqrt(sigma_mu_2);

        rho = sigma_mu_2/(sigma_mu_2+sigma_v_2);

        %%% regression, by Fuller and Battese
        theta = 1 - sigma_v/sigma_1;
        y_star = y - theta*Py;
        Z_star = Z - theta*PZ;

        delta = regress(y_star,Z_star);

        % The statistics parts are totally based on transfomed regression.
        %%% get rss
        y_estimate = Z*delta;
        residuals = y_star - Z_star*delta;
        rss = residuals'*residuals;

        %%% estimate variance
        s_2 = rss/(N*T-K-1);
        s = sqrt(s_2);

        %%% regression statistics
        V = s_2*(inv(Z_star.'*Z_star));
        std_err = diag((V).^(1/2));

        % Stata used z, not t. Perhpas it was because assuming s_2 is not
        % estimated in regression, but was estimated before (s_2 should be
        % equal to sigma_v_2 theoretically, but it's not numerically when
        % running in the program, because of potential numerial error)
        z = delta./std_err;
        p_value_z = 1 - (normcdf(abs(z)) - normcdf(-abs(z)));

        alpha = (1 - options.level/100);
        lower_bound = delta - abs(norminv(alpha/2)) * std_err;
        upper_bound = delta + abs(norminv(alpha/2)) * std_err;

        % Again, for the hypothesis test, since assumed s_2 is known.
        % So the F-stat reduces to chi2(K)-stat.
        [~,~,e,~] = regress(y_star, Z_star(:,1));
        RRSS = e.'* e;
        URSS = rss;

        chi2 = (RRSS - URSS) / s_2;
        p_value_chi2 = chi2cdf(chi2,K,'upper');

        % R^2  % see Stata pdf [xt] "Methods and formulas"
        temp = corr([QZ*delta, Qy]);
        R2_within = temp(1,2)^2;

        temp = corr([PZ*delta, Py]);
        R2_between = temp(1,2)^2;

        temp = corr([Z*delta, y]);
        R2_overall = temp(1,2)^2;

        % Summarizing Table
        Table = table;

        TableColumnNames = {model.depvar 'Coefficient' 'Std. err.'...
            'z' 'P>|z|' ['[',num2str(options.level),'% conf.'] 'interval]'};

        Table.(TableColumnNames{1}) = model.indepvar';
        Table.(TableColumnNames{2}) = delta;
        Table.(TableColumnNames{3}) = std_err;
        Table.(TableColumnNames{4}) = z;
        Table.(TableColumnNames{5}) = p_value_z;
        Table.(TableColumnNames{6}) = lower_bound;
        Table.(TableColumnNames{7}) = upper_bound;

        switch options.Table
            case 'Yes'
                disp(' ')
                disp(' ')
                disp(Table)
        end

        % results returned
        model.sigma_v_2 = sigma_v_2;
        model.sigma_v = sigma_v;
        model.sigma_mu = sigma_mu;
        model.sigma_mu_2 = sigma_mu_2;
        model.rho = rho;

        model.delta = delta;
        model.y_estimate = y_estimate;
        model.residuals = residuals;
        model.rss = rss;
        model.V = V;
        model.s_2 = s_2;
        model.s = s;

        model.std_err = std_err;
        model.z = z;
        model.p_value_z = p_value_z;
        model.chi2 = chi2;
        model.p_value_chi2 = p_value_chi2;

        model.R2_within = R2_within;
        model.R2_between = R2_between;
        model.R2_overall = R2_overall;
        model.Table = Table;

end


end

