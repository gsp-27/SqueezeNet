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

-- here will construct the entire squeezenet model as mentioned in the paper
local model = nn.Sequential()
-- first convolution layer output will be (96,16,16) 
model:add(nn.SpatialConvolution(3,96,3,3,1,1,1,1))
model:add(nn.SpatialBatchNormalization(96, 1e-5))
model:add(nn.ReLU(true))
model:add(nn.SpatialMaxPooling(2,2,2,2))

-- now have to add 8 fire modules
s1x1 = {32,32,64,64,96,96,128,128}
e1x1 = {64,64,128,128,192,192,256,256}
e3x3 = {64,64,128,128,192,192,256,256}

-- fire module 2 output (128,16,16)
model:add(FireModule(96,s1x1[1],e1x1[1],e3x3[1]))

-- Addition of simple block connection, so output of fire 2 will be sent to
-- output of fire 3 and the two will be added elementwise to add shortcut
-- connections
mlp = nn.ConcatTable()
-- fire module 3
mlp:add(FireModule(128, s1x1[2], e1x1[2], e3x3[2]))
mlp:add(nn.Identity())

model:add(mlp)
model:add(nn.CAddTable())
model:add(nn.ReLU(true))

-- fire module 4 output (256,16,16)
model:add(FireModule(128,s1x1[3],e1x1[3],e3x3[3]))
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- input size is reduced by 2, so (256,8,8)

mlp = nn.ConcatTable()
-- fire module 5
mlp:add(FireModule(256,s1x1[4],e1x1[4],e3x3[4]))
mlp:add(nn.Identity())

model:add(mlp)
model:add(nn.CAddTable())
model:add(nn.ReLU(true))

-- fire module 6 output (384,8,8)
model:add(FireModule(256,s1x1[5],e1x1[5],e3x3[5]))

mlp = nn.ConcatTable()
-- fire module 7
mlp:add(FireModule(384,s1x1[6],e1x1[6],e3x3[6]))
mlp:add(nn.Identity())

model:add(mlp)
model:add(nn.CAddTable())
model:add(nn.ReLU(true))

-- fire module 8 output (512,8,8)
model:add(FireModule(384,s1x1[7],e1x1[7],e3x3[7]))
model:add(nn.SpatialMaxPooling(2,2,2,2)) -- size reduced by 2 so (512,4,4)

-- fire module 9 output (512,4,4)
mlp = nn.ConcatTable()
mlp:add(FireModule(512,s1x1[8],e1x1[8],e3x3[8]))
mlp:add(nn.Identity())

model:add(mlp)
model:add(nn.CAddTable())
model:add(nn.ReLU(true))

model:add(nn.Dropout(0.7))
model:add(nn.SpatialConvolution(512,10,1,1))
model:add(nn.SpatialBatchNormalization(10, 1e-5))
model:add(nn.ReLU(true))
model:add(nn.SpatialAveragePooling(4,4,1,1))
model:add(nn.View(10))
-- return model

-- initialisation of weights as described in Kaiming He et.al paper
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(model)
-- print(model)
-- input = torch.randn(1,3,32,32)
-- scores = model:forward(input)
-- print(scores)

--print(model)
--input = torch.randn(1,3,32,32)
--scores = model:forward(input)
--print(scores)
return model
