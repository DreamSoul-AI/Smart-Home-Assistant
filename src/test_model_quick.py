import torch
import sys
sys.path.append('.')

from config import cfg, process_args
from model import make_model
from module import process_control

# 设置基本配置
cfg['control'] = {
    'data_name': 'SmartHome-hh105-all-LS005',
    'model_name': 'lstm'
}
cfg['tag'] = '0_SmartHome-hh105-all-LS005_lstm'
cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg['model_name'] = 'lstm'

# 处理控制配置
process_control()

# 设置必要的配置项
cfg['batch_size'] = 32
cfg['model']['lstm'] = {
    'hidden_size': 128,
    'num_layers': 2,
    'seq_len': 384,
    'label_len': 96,
    'pred_len': 96
}

print(f"设备: {cfg['device']}")
print(f"模型配置: {cfg['model']}")

try:
    # 创建模型
    print("\n创建模型...")
    model = make_model(cfg['model']).to(cfg['device'])
    print("✓ 模型创建成功")
    
    # 创建假数据测试前向传播
    print("\n测试前向传播...")
    batch_size = 32
    seq_len = 384
    features = 3
    
    fake_input = {
        'data': torch.randn(batch_size, seq_len + 96 + 96, features).to(cfg['device']),
        'attention_mask': torch.ones(batch_size, seq_len + 96 + 96, dtype=torch.bool).to(cfg['device']),
        'control_mask': torch.zeros(batch_size, seq_len + 96 + 96, dtype=torch.bool).to(cfg['device'])
    }
    
    print(f"输入数据形状: {fake_input['data'].shape}")
    
    # 运行模型
    with torch.no_grad():
        output = model(fake_input)
    
    print("✓ 前向传播成功")
    print(f"输出包含: {list(output.keys())}")
    if 'loss' in output:
        print(f"Loss值: {output['loss'].item():.4f}")
    if 'predicted_steps' in output:
        print(f"预测输出形状: {output['predicted_steps'].shape}")
    
    print("\n✅ 模型测试通过！可以正常运行训练。")
    
except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc() 