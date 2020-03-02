package actions;

import actions.util.ListNode;
import actions.util.TreeNode;

public class BitTigerTest {
    BitTiger test = new BitTiger();
    BitTiger2 test2 = new BitTiger2();

    @org.junit.Test
    public void testMethod() {
        BitTiger.LRUCache lru = new BitTiger.LRUCache(1);
        lru.put(2, 1);
        lru.get(2);
        lru.put(3, 2);
        int result = test.largestRectangleArea(new int[]{2, 1, 2});
        test.minWindow("ADOBECODEBANC", "ABC");
        test.search(new int[]{3, 1}, 3);
    }

    @org.junit.Test
    public void testMethod2() {
        test2.calculate2("3+2*2");
        test2.canFinish(2, new int[][]{{0, 1}, {1, 0}});
        test2.countPrimes(1500000);
        test2.isHappy(19);
        test2.fractionToDecimal(1, 6);
    }

    @org.junit.Test
    public void testMethodTreeNode() {
        TreeNode node0 = new TreeNode(0);
        TreeNode node1 = new TreeNode(1);
        TreeNode node2 = new TreeNode(2);
        TreeNode node3 = new TreeNode(3);
        TreeNode node4 = new TreeNode(4);
        TreeNode node5 = new TreeNode(5);
        TreeNode node6 = new TreeNode(6);
        TreeNode node7 = new TreeNode(7);
        TreeNode node8 = new TreeNode(8);
        node3.left = node5;
        node3.right = node1;
        node5.left = node6;
        node5.right = node2;
        node1.left = node0;
        node1.right = node8;
        node2.left = node7;
        node2.right = node4;
        test2.lowestCommonAncestor2(node3, node5, node1);
    }

    @org.junit.Test
    public void testMethodListNode() {
        ListNode node0 = new ListNode(0);
        test2.deleteNode(node0);
    }
}
