# 仅使用`numpy`编写的深度学习框架

- 在框架上基本模仿`pytorch`，用以学习神经网络的基本算法，如前向传播、反向传播、各种层、各种激活函数
- 采用面向对象的思想进行编程，思路较为清晰
- 想要自己**手写神经网络**的同学们可以参考一下
- 代码大体框架较为清晰，但不否认存在丑陋的部分，以及对于`pytorch`的拙劣模仿

## 项目介绍

- ### `MINST_recognition`:
		
	- 手写数字识别，使用`MINST`数据集
	
	- 训练30轮可以达到93%准确度，训练500轮左右达到95%准确度无法继续上升
	
- ### `RNN_sin_to_cos`:

	- 使用循环神经网络RNN，用$sin$的曲线预测$cos$的曲线
	
	- 目前仍有bug，无法正常训练

## 框架介绍

- 与框架有关的代码都放在了`mtorch`文件夹中

- ### 使用流程

  - 与`pytorch`相似，需要定义自己的神经网络、损失函数、梯度下降的优化算法等等

  - 在每一轮的训练中，先获取样本输入将其输入到自己的神经网络中获取输出。然后将**预测结果和期望结果**交给损失函数计算`loss`，并通过`loss`进行梯度的计算，最后通过优化器对神经网络的参数进行更新。

  - 结合代码理解更佳👇：

  - 以下是使用`MINST`数据集的**手写数字识别的主体代码**

  - ```python
  	# 定义网络 define neural network
  	class DigitModule(Module):
  	    def __init__(self):
  	        # 计算顺序就会按照这里定义的顺序进行
  	        sequential = Sequential([
  	            layers.Linear2(in_dim=ROW_NUM * COLUM_NUM, out_dim=16, coe=2),
  	            layers.Relu(16),
  	            layers.Linear2(in_dim=16, out_dim=16, coe=2),
  	            layers.Relu(16),
  	            layers.Linear2(in_dim=16, out_dim=CLASS_NUM, coe=1),
  	            layers.Sigmoid(CLASS_NUM)
  	        ])
  	        super(DigitModule, self).__init__(sequential)
  	
  	
  	module = DigitModule()  # 创建模型 create module
  	loss_func = SquareLoss(backward_func=module.backward)  # 定义损失函数 define loss function
  	optimizer = SGD(module, lr=learning_rate)  # 定义优化器 define optimizer
  	
  	
  	for i in range(EPOCH_NUM):  # 共训练EPOCH_NUM轮
  	    trainning_loss = 0  # 计算一下当前一轮训练的loss值，可以没有
  	    for data in train_loader:  # 遍历所有样本，train_loader是可迭代对象，保存了数据集中所有的数据
  	        imgs, targets = data  # 将数据拆分成图片和标签
  	        outputs = module(imgs)  # 将样本的输入值输入到自己的神经网络中
  	        loss = loss_func(outputs, targets, transform=True)  # 计算loss / calculate loss
  	        trainning_loss += loss.value
  	        loss.backward()  # 通过反向传播计算梯度 / calculate gradiant through back propagation
  	        optimizer.step()  # 通过优化器调整模型参数 / adjust the weights of network through optimizer
  	    if i % TEST_STEP == 0:  # 每训练TEST_STEP轮就测试一下当前训练的成果
  	        show_effect(i, module, loss_func, test_loader, i // TEST_STEP)
  	        print("{} turn finished, loss of train set = {}".format(i, trainning_loss))
  	```

- 接下来逐个介绍编写的类，这些类在`pytorch`中都有同名同功能的类，是仿照`pytorch`来的：

- ### `Module`类

	- 与`pytorch`不同，只能有一个`Sequential`类（序列），在该类中定义好神经网络的各个层和顺序，然后传给`Module`类的构造函数
	- **正向传播：**调用`Sequential`的正向传播
	- **反向传播：**调用`Sequential`的反向传播
	- 目前为止，这个类的大部分功能与`Sequential`相同，只是**套了个壳**保证与`pytorch`相同

- ### `lossfunction`

	- 有不同的`loss`函数，构造函数需要给他指定**自己定义的神经网络的反向传播函数**
	- 调用`loss`函数会返回一个`Loss`类的对象，该类记录了`loss`值。
	- 通过调用`Loss`类的`.backward()`方法就可以实现反向传播计算梯度
	- 内部机制：
		- 内部其实就是调用了**自己定义的神经网络的反向传播函数**
		- 也算是对于`pytorch`的一个**拙劣模仿，完全没必要**，直接通过`Module`调用就好

- ### 优化器：

  - 目前只实现了**随机梯度下降SGD**
  - 构造函数的参数是**自己定义的`Module`**。在已经计算过梯度之后，调用`optimizer.step()`改变`Module`内各个层的参数值
  - 内部机制：
    - 目前由于只有SGD一种算法，所以暂时也只是一个**拙劣模仿**
    - 就是调用了一下`Module.step()`，再让`Module`调用`Sequential.step()`，最后由`Sequential`调用内部各个层的`Layer.step()`实现更新
    - 梯度值在`loss.backward`的时候计算、保存在各个层中了

- ### `Layer`类

	- 有许多不同的层

	- #### 共性

		- **前向传播**：
			- 接受一个输入进行前向传播计算，输出一个输出
			- 会将输入保存起来，在反向传播中要用
		- **反向传播**：
			- 接受**前向传播的输出的梯度值**，计算**自身参数（如Linear中的w和b）的梯度值**并保存起来
			- 输出值为**前向传播的输入的梯度值**，用来让上一层（可能没有）继续进行反向传播计算
			- 这样不同的层之间就可以进行任意的拼装而不妨碍前向传播、反向传播的进行了
		- **`.step`方法**
			- 更新自身的参数值（也可能没有，如激活层、池化层）

	- #### `Sequential`类

		- 这个类也是继承自`Layer`，可以当作一层来使用

		- 它把多个层按照顺序拼装到一起，在前向、反向传播时按照顺序进行计算

		- 结合它的`forward`、`backward`方法来理解：

			- ```python
				def forward(self, x):
				    out = x
				    for layer in self.layers:
				        out = layer(out)
				    return out
				
				def backward(self, output_gradient):
				    layer_num = len(self.layers)
				    delta = output_gradient
				    for i in range(layer_num - 1, -1, -1):
				        # 反向遍历各个层, 将期望改变量反向传播
				        delta = self.layers[i].backward(delta)
				
				def step(self, lr):
				    for layer in self.layers:
				        layer.step(lr)
				```

			

	- ### `RNN`类：循环神经网络层

		- 继承自`Layer`，由于内容比较复杂故单独说明一下

		- `RNN`内部由一个**全连接层`Linear`**和一个**激活层**组成

		- #### 前向传播

			- ```python
				    def forward(self, inputs):
				        """
				        :param inputs: input = (h0, x) h0.shape == (batch, out_dim) x.shape == (seq, batch, in_dim)
				        :return: outputs: outputs.shape == (seq, batch, out_dim)
				        """
				        h = inputs[0]  # 输入的inputs由两部分组成
				        X = inputs[1]
				        if X.shape[2] != self.in_dim or h.shape[1] != self.out_dim:
				            # 检查输入的形状是否有问题
				            raise ShapeNotMatchException(self, "forward: wrong shape: h0 = {}, X = {}".format(h.shape, X.shape))
				
				        self.seq_len = X.shape[0]  # 时间序列的长度
				        self.inputs = X  # 保存输入，之后的反向传播还要用
				        output_list = []  # 保存每个时间点的输出
				        for x in X:
				            # 按时间序列遍历input
				            # x.shape == (batch, in_dim), h.shape == (batch, out_dim)
				            h = self.activation(self.linear(np.c_[h, x]))
				            output_list.append(h)
				        self.outputs = np.stack(output_list, axis=0)  # 将列表转换成一个矩阵保存起来
				        return self.outputs
				```

		- #### 反向传播

			- ```python
				def backward(self, output_gradient):
				    """
				    :param output_gradient: shape == (seq, batch, out_dim)
				    :return: input_gradiant
				    """
				    if output_gradient.shape != self.outputs.shape:
				        # 期望得到(seq, batch, out_dim)形状
				        raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
				                                           "{}".format(self.outputs.shape, output_gradient.shape))
				
				    input_gradients = []
				    # 每个time_step上的虚拟weight_gradient, 最后求平均值就是总的weight_gradient
				    weight_gradients = np.zeros(self.linear.weights_shape())
				    bias_gradients = np.zeros(self.linear.bias_shape())
				    batch_size = output_gradient.shape[1]
				
				    # total_gradient: 前向传播的时候是将x, h合成为一个矩阵，所以反向传播也先计算这个大矩阵的梯度再拆分为x_grad, h_grad
				    total_gradient = np.zeros((batch_size, self.out_dim + self.in_dim))
				    h_gradient = None
				    
				    # 反向遍历各个时间层，计算该层的梯度值
				    for i in range(self.seq_len - 1, -1, -1):
				        # 前向传播顺序: x, h -> z -> h
				        # 所以反向传播计算顺序：h_grad -> z_grad -> x_grad, h_grad, w_grad, b_grad
				
				        # %%%%%%%%%%%%%%计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
				        # h_gradient = (output_gradient[i] + total_gradient[:, 0:self.out_dim]) / 2
				        # %%%%%%%%%%%%%%不计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
				        #  计算h_grad: 这一时间点的h_grad包括输出的grad和之前的时间点计算所得grad两部分
				        h_gradient = output_gradient[i] + total_gradient[:, 0:self.out_dim]  
				
				        # w_grad和b_grad是在linear.backward()内计算的，不用手动再计算了
				        z_gradient = self.activation.backward(h_gradient)  # 计算z_grad
				        total_gradient = self.linear.backward(z_gradient)  # 计算x_grad和h_grad合成的大矩阵的梯度
				
				        # total_gradient 同时包含了h和x的gradient, shape == (batch, out_dim + in_dim)
				        x_gradient = total_gradient[:, self.out_dim:]
				
				        input_gradients.append(x_gradient)  
				        weight_gradients += self.linear.gradients["w"]
				        bias_gradients += self.linear.gradients["b"]
				
				    # %%%%%%%%%%%%%%%%%%计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
				    # self.linear.set_gradients(w=weight_gradients / self.seq_len, b=bias_gradients / self.seq_len)
				    # %%%%%%%%%%%%%%%%%%不计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
				    self.linear.set_gradients(w=weight_gradients, b=bias_gradients)  # 设置梯度值
				    
				    list.reverse(input_gradients)  # input_gradients是逆序的，最后输出时需要reverse一下
				    print("sum(weight_gradients) = {}".format(np.sum(weight_gradients)))
				    
				    # np.stack的作用是将列表转变成一个矩阵
				    return np.stack(input_gradients), h_gradient
				```