local TVLoss, parent = torch.class('nn.TotalVariation', 'nn.Module')


function TVLoss:__init(strength, tv_weight, tanh_constant)
  parent.__init(self)
  self.strength = strength or 0
  self.tv_weight = tv_weight or 0
  self.tanh_constant = tanh_constant or 1
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
  self.abs_diff = torch.Tensor()
  self.x_weight = torch.Tensor()
  self.y_weight = torch.Tensor()
end


function TVLoss:updateOutput(input)
  local N, C = input:size(1), input:size(2)
  local H, W = input:size(3), input:size(4)
  if self.tv_weight==1 then
    local amp = torch.zeros(N,1,H,W):cuda()
    amp:copy(input[{{},{-1},{},{}}]):mul(self.tanh_constant)
    self.x_weight:resize(N, 1, H - 1, W - 1)
    self.y_weight:resize(N, 1, H - 1, W - 1)
    self.x_weight:copy(amp[{{}, {}, {1, -2}, {1, -2}}])
    self.x_weight:add(-1, amp[{{}, {}, {1, -2}, {2, -1}}])
    self.y_weight:copy(amp[{{}, {}, {1, -2}, {1, -2}}])
    self.y_weight:add(-1, amp[{{}, {}, {2, -1}, {1, -2}}])
    torch.exp(self.x_weight, -torch.abs(self.x_weight))
    torch.exp(self.y_weight, -torch.abs(self.y_weight))
    torch.repeatTensor(self.x_weight, 1,C-1,1,1)
    torch.repeatTensor(self.y_weight, 1,C-1,1,1)
    input = input[{{},{1,-2},{},{}}]
    C = C-1
  end
  self.x_diff:resize(N, C, H - 1, W - 1)
  self.y_diff:resize(N, C, H - 1, W - 1)
  self.abs_diff:resize(N, C, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {}, {2, -1}, {1, -2}}])
  if self.tv_weight == 1 then
    self.x_diff:cmul(self.x_weight)
    self.y_diff:cmul(self.y_weight)
  end
  -- self.abs_diff = self.abs_diff:zero()+1
  self.abs_diff = torch.sqrt(torch.pow(self.x_diff,2)+torch.pow(self.y_diff,2))+1e-6
  self.output:resizeAs(input):copy(input)
  -- self.output[{{}, {}, {1, -2}, {1, -2}}]:add(self.strength*self.abs_diff)
  return self.output
end


-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  local C = input:size(2)
  if self.tv_weight==1 then
    C = C-1
  end
  self.gradInput:resizeAs(input):zero()
  self.gradInput[{{}, {1,C}, {1, -2}, {1, -2}}]:add(torch.add(self.x_diff,self.y_diff):cdiv(self.abs_diff))
  self.gradInput[{{}, {1,C}, {1, -2}, {2, -1}}]:add(-1, torch.cdiv(self.x_diff,self.abs_diff))
  self.gradInput[{{}, {1,C}, {2, -1}, {1, -2}}]:add(-1, torch.cdiv(self.y_diff,self.abs_diff))
  self.gradInput:mul(self.strength)-- 
  -- self.gradInput[{{}, {1,C}, {}, {}}]:cmul(gradOutput)
  self.gradInput[{{}, {1,C}, {}, {}}]:add(gradOutput)
  return self.gradInput
end

