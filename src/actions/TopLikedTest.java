package actions;

public class TopLikedTest {
    TopLiked test = new TopLiked();

    @org.junit.Test
    public void testMethod() {
        test.maximalRectangle(new char[][]{{'0', '1'}, {'1', '0'}});
        test.isMatch("aa", "a*");
    }
}
