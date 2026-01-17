const API_URL = import.meta.env.VITE_API_URL || '';

async function request(endpoint, options = {}) {
  const url = endpoint.startsWith('http') ? endpoint : `${API_URL}${endpoint}`;
  
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.message || `Request failed: ${response.status}`);
  }
  
  return response.json();
}

export async function createResearch(data) {
  return request('/api/research', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getResearch(taskId) {
  return request(`/api/research/${taskId}`);
}

export async function cancelResearch(taskId) {
  return request(`/api/research/${taskId}`, {
    method: 'DELETE',
  });
}

export const SCENARIOS_DATA = [
  {
    id: "fact-check",
    name: "事实核查",
    icon: "🔍",
    description: "验证信息真实性，交叉比对多个信源",
    prompt_template: "请核实以下信息的真实性并溯源：\n\n{user_input}\n\n要求：\n1. 查找原始出处\n2. 交叉验证至少3个独立信源\n3. 明确指出信息是否属实\n4. 如有错误，说明实际情况",
    suggested_tools: ["search", "visit", "google_scholar"]
  },
  {
    id: "competitor-analysis",
    name: "竞品分析",
    icon: "📊",
    description: "深度对比竞争产品，生成分析报告",
    prompt_template: "请对以下产品/公司进行深度竞品分析：\n\n{user_input}\n\n报告应包括：\n1. 产品功能对比\n2. 定价策略\n3. 技术栈差异\n4. 市场定位\n5. 优劣势总结",
    suggested_tools: ["search", "visit"]
  },
  {
    id: "literature-review",
    name: "学术综述",
    icon: "📚",
    description: "检索学术论文，生成文献综述",
    prompt_template: "请对以下研究主题进行学术文献综述：\n\n{user_input}\n\n要求：\n1. 检索近3年相关论文\n2. 总结主要研究方向\n3. 归纳关键发现\n4. 指出研究空白和未来方向",
    suggested_tools: ["google_scholar", "visit", "search"]
  },
  {
    id: "due-diligence",
    name: "投前尽调",
    icon: "💰",
    description: "投资前尽职调查，风险评估",
    prompt_template: "请对以下公司进行投资尽职调查：\n\n{user_input}\n\n调查范围：\n1. 公司背景和团队\n2. 融资历史\n3. 业务模式和市场规模\n4. 技术壁垒\n5. 风险因素",
    suggested_tools: ["search", "visit"]
  },
  {
    id: "compliance-review",
    name: "合规审查",
    icon: "⚖️",
    description: "法规检索，合规性建议",
    prompt_template: "请针对以下场景进行合规性审查：\n\n{user_input}\n\n审查内容：\n1. 相关法律法规\n2. 监管要求\n3. 合规风险点\n4. 整改建议",
    suggested_tools: ["search", "visit", "google_scholar"]
  }
];

export async function getScenarios() {
  // Return the synchronous data for now
  return SCENARIOS_DATA;
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/api/upload`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error('Upload failed');
  }
  
  return response.json();
}

export async function getConfig() {
  return request('/api/config');
}