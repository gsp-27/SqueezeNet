require 'nn'

-- s, e1, e2 should be filter numbers
local function FireModule(nInputPlane, s, e1, e2)
	-- this will form the squeeze layer
	local sNet = nn.Sequential()
	sNet:add(nn.SpatialConvolution(nInputPlane,s,1,1))
	sNet:add(nn.ReLU(true))

	-- Construction of expand layer
	local mlp = nn.DepthConcat(2)
	mlp:add(nn.SpatialConvolution(s,e1,1,1))
	mlp:add(nn.SpatialConvolution(s,e2,3,3))
	sNet:add(mlp)
	sNet:add(nn.ReLU(true))
	return sNet
end

local model = FireModule(3, 5, 4, 6)
print(model)
input = torch.randn(10,3,7,7)
print(model:forward(input))