import ReactFlow, {
  Background,
  useNodesState,
  useEdgesState,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { useResearchStore } from '../../stores/researchStore';
import { useMemo, useEffect } from 'react';
import QuestionNode from './QuestionNode';
import RolloutNode from './RolloutNode';
import AnswerNode from './AnswerNode';
import styles from './FlowGraph.module.css';

const nodeTypes = {
  question: QuestionNode,
  rollout: RolloutNode,
  answer: AnswerNode,
};

export default function FlowGraph() {
  const { question, rollouts, status, result } = useResearchStore();
  
  // We use useNodesState/useEdgesState from reactflow but update them via useMemo
  // In a real interactive graph we'd sync them. For visualization only, simple mapping works.
  
  const { nodes, edges } = useMemo(() => {
    const nodes = [];
    const edges = [];
    
    // Question Node
    if (question) {
      nodes.push({
        id: 'question',
        type: 'question',
        position: { x: 300, y: 0 },
        data: { question },
      });
    }
    
    // Rollout Nodes
    rollouts.forEach((rollout, index) => {
      // Calculate position to center them
      const totalWidth = rollouts.length * 250;
      const startX = 300 - (totalWidth / 2) + 125;
      const xPos = startX + index * 250;
      
      nodes.push({
        id: `rollout-${rollout.id}`,
        type: 'rollout',
        position: { x: xPos, y: 150 },
        data: rollout,
      });
      
      if (question) {
        edges.push({
          id: `e-question-rollout-${rollout.id}`,
          source: 'question',
          target: `rollout-${rollout.id}`,
          animated: rollout.status === 'running',
          style: { stroke: '#555' },
        });
      }
    });
    
    // Answer Node
    // Show if at least one rollout started
    if (rollouts.length > 0) {
       nodes.push({
        id: 'answer',
        type: 'answer',
        position: { x: 300, y: 320 },
        data: { status, result },
      });
      
      rollouts.forEach((rollout) => {
        edges.push({
          id: `e-rollout-${rollout.id}-answer`,
          source: `rollout-${rollout.id}`,
          target: 'answer',
          animated: status === 'running' && rollout.status === 'completed',
          style: { stroke: rollout.status === 'completed' ? 'var(--accent-cyan)' : '#333' },
        });
      });
    }
    
    return { nodes, edges };
  }, [question, rollouts, status, result]);
  
  if (!question && rollouts.length === 0) return null;
  
  return (
    <div className={styles.panel}>
      <div className={styles.title}>─ 多路径探索</div>
      <div className={styles.graphContainer}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          attributionPosition="bottom-left"
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#333" gap={16} size={1} />
        </ReactFlow>
      </div>
      <div className={styles.hint}>点击节点查看详情</div>
    </div>
  );
}
