# 创建交货单手册

Created: July 30, 2023 8:02 PM

操作手册

交货单流程

目录
1	总览	4
1.1	总体描述	4
1.2	事务代码	4
2	操作步骤描述	5
2.1	事务一：创建上海西美/陕西新西美交货单	5
2.2	事务二：修改上海西美/陕西新西美交货单	16
2.3	事务三：查询上海西美/陕西新西美交货单	23
 
本文件的相关信息
版本概要
版本号	作者	日期	变更记录	修订
01	杨辉	2017/4/17	初稿	1.0

审批信息
姓名	角色	批准日期	签名

1	总览
基本图标及含义
符号	名称	涵义

键入	功能与键盘上的” ENTER”相同

事务处理代码	输入事务处理代码的地方

保存	保存键

返回	返回前屏幕

退出	返回交易的主屏幕
取消	返回主菜单

页键	(第一页、上一页、下一页、最后一页)

1.1	总体描述
本文档介绍对于交货单的创建这个功能，是针对西美交货单创建一些特殊要求做的定制开发。
可实现：1.销售订单的合并发货 2.销售订单拆单发货 3.销售订单按单发货（整单）

(各位亲，第一遍看到这个文档的时候，请别单看图片，一定要仔细看看文字说明哟！)

1.2	事务代码
事务代码	菜单路径
ZSDU001	无

2	操作步骤描述

- 事务代码输入的位置
事务代码	ZSDU001

2.1	事务一：创建上海西美/陕西新西美交货单
事务代码	ZSDU001

步骤1：通过事务代码ZSDU001，进入如下界面：

步骤2：点击交货单创建，则会进入到如下界面：

在这个界面主要涉及到6个功能按钮键。为实现交货单的创建，将做详细说明。
1：查询按钮。初始界面如下：

点击查询按钮，则会进入到如下界面。

其实这一步所要做的事情很简单，就是把你需要交货的销售订单项目，进行筛选，作为创建交货单的备选数据！我在client300用10000107这个销售订单举例：输入销售订单号和计划交货日期，则会把满足交期的销售订单项目查询出，选择需要交货的项目，确定即可！

选中后，点击确定，则会到如下界面。

强调：针对合并交货（即多张销售订单的项目合并到同一张交货单的情况），这里除了满足上海西/陕西新西美提出的必须要满足的条件，还需要满足一些SAP标准的合并交货的条件要求，汇总即为截图中红框里面的条件。

对于图片中的订单组合和完整交货，为销售订单中的如下标识。

只有当销售订单里面的这些条件都相同的时候，才能合并发货。

当把所有的需要交货的销售订单项目都选到选择订单信息界面的时候，可以用红框中的 按钮，对已经选择的行项目进行调整。

在这个订单选择界面，有一列建议交货数量，用户可以根据实际需求，对交货数量进行调整。
当然，这一列的建议交货数量，在物料启用了ATP检查的时候，这一列的数量也是考虑了ATP库存的数量，所以在创建交货单的时候，这一列的数据，修改的时候，只能改小，不能改大，改大时候会有信息提醒，因为当前的建议交货数量，已经是最大的ATP库存使用量。

2.模拟按钮

点击模拟按钮之后，程序会根据已经选择的销售订单数据，计算出如下图，一些汇总信息和订单抬头信息，订单行项目数据。

注意：因为交货单的抬头信息，包括毛净体等等信息，都是严格按照你选择的订单数据模拟出来的数据，所以，当你对第一步中的内容（订单选择数据）做了任何调整之后（包括删除行项目，修改交货数量等），都需要重新点击模拟按钮，重新计算。切记！

模拟完成之后，在行项目信息页签，这里面的箱数，单箱毛净体、总毛净体、都是来源于销售订单数据，当由于实际业务，这些数据不完整或不正确的时候，1、我们首先考虑到的是物料主数据不完整或不正确，需要找到相对应的部门和人员，补齐物料主数据。2、销售订单数据不完整或不正确，需要找到相对应的业务人员补齐销售订单数据。对应的数据补齐之后，我们在来从头开始，做这个交货单。3、对于一些特殊的业务，比如量刃具，无论是装数、还是毛净体都无法在主数据固定下来,也可以在这里创建交货单的时候，直接录入箱数、毛净体等。

涉及到跨公司销售(4000,5000)下单,1000,2000发货的情况下,
报关金额取得是内部结算成本的价格,
如未涉及,则交货金额等于报关金额.
交货金额对应的是:最终客户销售订单的交货金额
(涉及到跨公司销售(4000,5000)下单,1000,2000发货的情况)

报关金额是:针对该交货单实际报关时的总金额

3、信用证导入
支付方式为信用证，首先查询信用证信息是否导入SAP。
可通过事物代码ZSDF009,进入如下界面：

可通过输入公司代码，再输入交货单对应的销售订单号，即可查询对应的信用证号和证序列号。

查询到相应的信用证号或者证序号，即可导入信用证信息。

导入后，即可带出如下界面：

4、复制抬头

已存在的交货单抬头信息可以复制到当前交货单。前提是同一个售达方。
出现以下界面：

5、创建按钮
根据需要交货的订单数据，仔细核对抬头信息数据和行项目数据，确认无误后，点击创建按钮。即可点击创建按钮创建发货单。

6、清除按钮
点“重置”，意味一个交货单创建完成之后，清除所有数据，接着做下一单数据。

2.2	事务二：修改上海西美/陕西新西美交货单
步骤1：通过事务代码ZSDU001，进入如下界面：

步骤2.输入交货单号, 回车

1．更改抬头数据。

我们创建交货单时抬头信息里面的数据，主要在两个页签里面，如下图：

对于创建好的交货单，抬头信息出现错误的情况，直接在这两个页签修改。

2.更改项目的毛净体，长宽高数据。
双击具体行项目

3、添加行项目

填入计划交期、销售订单、行项目号、点击   即可导入
错误谨记：许多的同事在交货单上添加行项目是以直接输入物料号和数据的形式实现的，这种方法是绝对错误的。我们必须以上图的方式添加交货单行项目。因为我们的任何一笔交货都是有销售订单参考，行项目信息会根据主数据自动带出，当主数据不齐时，需手动补齐数据。如图，添加行项目后自动带出的内容。

4、删除行项目

5.修改单位
如果发现行项目单位有误,修改该物料,事务代码为:MM02,进入如下界面:

点"附加数据”

添加需要增加的单位:

返回到交货单修改,修改需要修改的单位:

以上修改单位属于报关单位应急处理方式

2.3	事务三：查询上海西美/陕西新西美交货单
步骤1：通过事务代码ZSDU001，进入如下界面：

步骤2:输入交货单号进行查询即可.