print '==> downloading dataset'

-- Here we download dataset files. 

-- Note: files were converted from their original LUSH format
-- to Torch's internal format.

-- The SVHN dataset contains 3 files:
--    + train: training data
--    + test:  test data

tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

if not paths.dirp('mnist.t7') then
   os.execute('wget ' .. tar)
   os.execute('tar xvf ' .. paths.basename(tar))
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'

----------------------------------------------------------------------
print '==> loading dataset'

-- We load the dataset from disk, it's straightforward

trainData = torch.load(train_file,'ascii')
testData = torch.load(test_file,'ascii')

print('Training Data:')
print(trainData)
print()

print('Test Data:')
print(testData)
print()

----------------------------------------------------------------------
print '==> visualizing data'

-- Visualization is quite easy, using itorch.image().
if itorch then
   print('training data:')
   itorch.image(trainData.data[{ {1,256} }])
   print('test data:')
   itorch.image(testData.data[{ {1,256} }])
end

require 'nn'
require 'image'
require 'cunn'
require 'cutorch'
require 'optim'

local model = nn.Sequential()

-- input: (batch x) 1 x 32 x 32

-- in, out, width, height, stridew, strideh, padw, padh
model:add(nn.SpatialConvolution(1, 20, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
-- (batch x) 20 x 32 x 32


-- width, height, stridew, strideh
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- (batch x) 20 x 16 x 16

model:add(nn.SpatialConvolution(20, 20, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
-- (batch x) 20 x 16 x 16 , no change

model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- (batch x) 20 x 8 x 8

model:add(nn.SpatialConvolution(20, 10, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
-- (batch x) 10 x 8 x 8

model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- (batch x) 10 x 4 x 4

model:add(nn.Reshape(160))
model:add(nn.Linear(160, 10))
-- (batch x) 10

model:add(nn.LogSoftMax())

local criterion = nn.ClassNLLCriterion()

-- TODO: does this use cudnn, or custom cuda implementations? cudnn is
-- nvidia-made and will be faster than anything some random scientists wrote
model:cuda()
criterion:cuda()

-- TODO: add batch norm
-- TODO: add dropout

local batchSize = 50
parameters, gradParameters = model:getParameters()
local optim_state = {learningRate = 0.01, alpha = 0.95}

local womp = trainData.data:size(1)
for t = 1,womp,batchSize do
    xlua.progress(t, womp)
    shuffle = torch.randperm(womp)
    
    local this_batch_size = (math.min(t+batchSize-1,womp) - t) + 1
    local batch_input = torch.Tensor(this_batch_size, 1, 32, 32)
    local batch_target = torch.Tensor(this_batch_size)

    local offset = 1
    for i = t,(t + this_batch_size - 1) do
        -- load new sample
        local input = trainData.data[shuffle[i]]
        local target = trainData.labels[shuffle[i]]
        batch_input[{offset, {}, {}, {}}] = input
        batch_target[{offset}] = target

        offset = offset + 1
    end

    batch_input = batch_input:float():cuda()
    batch_target = batch_target:float():cuda()

    -- TODO: make the batch with batch_input
    local feval = function(x)
        if x ~= parameters then
            parameters:copy(x)
        end


        -- reset gradients
        gradParameters:zero()
        local output = model:forward(batch_input)
        local err = criterion:forward(output, batch_target)

        dferr = criterion:backward(output, batch_target)
        model:backward(batch_input, dferr)

        return err, gradParameters
    end
    local _, loss = optim.rmsprop(feval, parameters, optim_state)
    print(loss)
end
-- TODO: how does one ensure that this is using cudnn? we aren't even using the
-- gpu at the moment, nevermind the best gpu implementation



