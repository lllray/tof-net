function C = GetCorrelationMatrix(f,d)

    c = 3e8;
    nf = numel(f);
    nd = numel(d);
    C = zeros(nf, nd);
    z = 0; %2.00;
    for fi = 1:nf
        for di = 1:nd
            C(fi,di) = exp( 1i * 4*pi*f(fi)/c * (z+d(di)/2) );
        end
    end
    C  = [real(C); imag(C)]; %Euler's formula: e^(i phi) = cos(phi) + i sin(phi)

end

% h = C*i;
% h = h';
% h0  = h(:,1:221);
% h90 = h(:,222:end);
% hh = h0 ./ abs(h0+1i*h90); % normalize amp of phase meas
% if verbose
%     hold on; grid on; 
%     for j = 1:size(hh,1)
% %         plot3(1:221, j*ones(1,221), hh(j,:), 'LineWidth', 2); view(0,0);
%         plot(hh(j,:));
%     end
%     xlabel('Frequecies (MHz)');
%     ylabel('Raw values (normalized)');
%     title('simulated ToF raw measurements');
% %     legend(legendInfo);
%     hold off
% end
% 
% %% show C
% figure(2);
% C0 = C(1:221,:);
% imagesc(C0); xlabel('Time (ns)'); ylabel('Frequencies (MHz)');
% ax = gca;
% ax.XTick = 0:30:200;
% % ax.YTick = [-1 -0.5 0 0.5 1];
% ax.XTickLabel = {'0','10','20','30','40','50','60'};
% ax.YTickLabel = {'10','20','30','40','50','60','70','80','90','100','110'};