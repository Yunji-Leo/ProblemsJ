package actions;

import actions.util.TreeNode;

public class TopLikedTest {
    TopLiked test = new TopLiked();

    @org.junit.Test
    public void testMethod() {
        test.maximalRectangle(new char[][]{{'0', '1'}, {'1', '0'}});
        test.isMatch("aa", "a*");
    }

    @org.junit.Test
    public void testTree() {
        TreeNode n1 = new TreeNode(1);
        TreeNode n2 = new TreeNode(2);
        TreeNode n3 = new TreeNode(3);
        TreeNode n4 = new TreeNode(4);
        TreeNode n5 = new TreeNode(5);
        n1.left = n2;
        n1.right = n3;
        n3.left = n4;
        n3.right = n5;
        TopLiked.Codec c = new TopLiked().new Codec();
        String s = c.serialize(n1);
        TreeNode n = c.deserialize(s);

    }
}
