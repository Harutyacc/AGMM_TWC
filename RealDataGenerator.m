function [dataMod, DataUnmod] = RealDataGenerator(Num_samples, clusters, Phi, S, Sigma)
%     S = 0.8
%     Sigma = sqrt(S^2/(2*Rice_K))
    switch clusters
        case 4
            temp = randi(2, 2, Num_samples)*2 - 3;
        case 16
            temp=randi(2,2,Num_samples)*2-3;
            temp = 2 * temp.*(randi([1, 2], 2, Num_samples)-0.5);
        otherwise
            disp('undisired numbers!');
    end
    
    Symbol_Tx = (temp(1, :) + 1j*temp(2, :))./sqrt(2);
    
    noise = randn(1,Num_samples)*Sigma + 1j*randn(1,Num_samples)*Sigma;
    % Symbol_mod = Symbol_Tx.*S.*exp(1j*Phi);    
    Symbol_rx = Symbol_Tx.*S.*exp(1j*Phi) + noise;
    rx_mle = S.*exp(1j*Phi) + noise;    
    dataMod = [real(Symbol_rx); imag(Symbol_rx)]';
    DataUnmod = [real(rx_mle); imag(rx_mle)]';
end
